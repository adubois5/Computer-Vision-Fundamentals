# %%
import os
import numpy as np
import torch
import umap
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %%
##################################
# Change these
p = 3  # [3, 8, 16]
training = False
TRAIN_DATA_PATH = '/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/train/'
EVAL_DATA_PATH = '/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/test/'
LOAD_PATH = f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/model_{p}.pt"
OUT_PATH = '/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/exp/'
##################################

# %%
class DimReducerBuilder(Dataset):
    def __init__(self, path, option):
        self.path = path
        self.image_list = [f for f in os.listdir(path) if f.endswith('.png')]
        self.label_list = [int(f.split('_')[0]) for f in self.image_list]
        self.len = len(self.image_list)
        if option == "bw":
            self.aug = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            self.aug = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fn = os.path.join(self.path, self.image_list[index])
        x = Image.open(fn).convert('RGB')
        x = self.aug(x)
        
        # Flatten and normalize the image data:
        img_vec = torch.reshape(x, shape=(x.shape[0], -1))
        img_vecn = img_vec / torch.norm(img_vec, dim=1, keepdim=True)
        # img_vecn = img_vec / img_vec.shape[1]
        return {'img_vecn': img_vecn.squeeze(), 'y': self.label_list[index]}


# %% [markdown]
# ### Task 1: PCA

# %%
from sklearn.decomposition import PCA

# %%
def get_id_of_nearest_embedding(training_set, probe_embedding, num_classes=30, num_samples=21):
    index_to_class_id = np.repeat(np.arange(num_classes), num_samples)
    training_set = np.reshape(training_set.copy(), newshape=(training_set.shape[0]*training_set.shape[1], -1))
    # We first calculate the euclidean distance between the probe and all trained embeddings:
    distances = np.linalg.norm(training_set - probe_embedding, axis=1)
    nearest_neighbor_idx = np.argmin(distances)
    
    # Then we return the index with the smallest distance
    return index_to_class_id[nearest_neighbor_idx]

# %%
def get_PCA_vecs(img_vec, p, speedup=True):
    overall_mean = np.mean(img_vec, axis=0)
    
    # Substract by the mean:
    img_meaned = img_vec - overall_mean
    
    # PCA: 
    # Return p eigenvectors from the covariance matrix:
    # Calculate the covariance matrix:
    if speedup == False:
        print("IMg_meanign", img_meaned.shape)
        _, _, Vt = np.linalg.svd(img_meaned @ img_meaned.T, full_matrices=False) 

        W_p = img_meaned.T @ Vt

        # Normalize the principal components
        W_p = W_p / np.linalg.norm(W_p, axis=1, keepdims=True)        
        
        # Extract the top-p right singular vectors (principal components)
        W_phat = W_p[:, :p]  # Shape is (CHW, p)
    else:
        C = img_meaned @ img_meaned.T # Shape is (B, B)
        
        # Take the eig vecs of this, they are already in reverse order:
        _, eigvec_c = np.linalg.eigh(C)
        
        W_p = img_meaned.T @ eigvec_c  # Shape is (CHW, p)
        W_phat = W_p / np.linalg.norm(W_p, axis=1, keepdims=True) # Normalized
        W_phat = W_phat[:, -p:]
    
    # Project the image vectors into the p dimensional space
    pca = img_meaned @ W_phat
    return pca

# %% [markdown]
# ### Task 1: LDA

# %%
def calculate_global_parameters_for_LDA(train_loader, num_classes=30, feature_dim=4096):  
    num_classes = 30
    overall_mean = 0
    class_means = np.zeros((num_classes, feature_dim))
    num_samples = 0
    class_sample_counts = np.zeros((num_classes))

    for batch in train_loader:
        img_vecn = batch["img_vecn"].numpy()
        labels = batch["y"] # Shape is (B,)
        
        # Accumulate means:
        overall_mean += np.sum(img_vecn, axis=0) # shape is: (4096) -> (CHW)
        num_samples += img_vecn.shape[0]

        # Set up the dataset wide information required
        for label in np.unique(labels):
            # idx of class_means = label + 1
            class_samples = img_vecn[labels == label]
            class_sample_counts[label - 1] += class_samples.shape[0]
            class_means[label - 1] += np.sum(class_samples, axis=0)

    # Finalize the means:
    overall_mean = np.array(overall_mean / num_samples)  # Shape: (4096,)
    for label in range(num_classes):
        class_means[label] /= class_sample_counts[label]  # Shape: (4096,)

    # Compute S_B, the outer product for the between-class scatter:
    S_B = np.zeros((feature_dim, feature_dim)) # Shape is (4096, 4096)
    for i in range(num_classes):
        mean_diff = np.expand_dims(class_means[i] - overall_mean, axis=1) # Shape is (4096, 1)
        S_B += mean_diff @ mean_diff.T

    # Compute Within-Class Scatter Matrix (S_W)
    S_W = np.zeros((feature_dim, feature_dim)) # Shape is (4096, 4096)
    for batch in train_loader:
        img_vecn = batch["img_vecn"].numpy()
        labels = batch["y"] # Shape is (B,)
        
        # Set up the dataset wide information required
        for label in np.unique(labels):
            # idx of class_means = label + 1
            class_samples = img_vecn[labels == label]
            
            for sample in class_samples:
                diff = np.expand_dims((sample - class_means[label - 1]), axis=1)  # Shape: (4096, 1)
                S_W += diff @ diff.T  # Outer product shape is (4096, 4096)
    return S_B, S_W

# %%
def get_projection_matrix(S_W, S_B, p, lda_option, num_labels=30):
    if lda_option == "YUYANG":
        # Retain eigvecs where the values are not close to 0
        eigvals, eigvecs = np.linalg.eigh(S_B)
        idx = eigvals > 1e-6  # Filter eigenvalues
        top_eigvals, top_eigvecs = eigvals[idx], eigvecs[:, idx]
        
        # Normalize the eigenvectors:
        sb_eigvecn = top_eigvecs / np.linalg.norm(top_eigvecs, axis=1, keepdims=True) # Shapes is (CHW, K_Y)
        
        eig_val_mat = np.diag(top_eigvals) #np.eye(num_labels - 1) * top_eigvals
        # We can then construct a low dimensional projection of S_B with sb_eigvecn
        D_B = np.sqrt(np.linalg.inv(eig_val_mat))
        Z = np.dot(sb_eigvecn, D_B)
        
        # Use eigendecomp to diagonalize Z
        _, U = np.linalg.eigh(Z.T @ S_W @ Z)
        # Get the top eigenvectors and normalize them
        U_top = U[:, -p:]
        U_topn = U_top / np.linalg.norm(U_top, axis=1, keepdims=True)
        
        # Generate the projection matrix
        proj_mat = (U_topn.T @ Z.T).T
    else:
        # LDA Optimization
        # Get the eigenvalue/vectors of S_W^-1 S_B
        _, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

        # Get the top eigvecs since np already sorts them
        proj_mat = eigvecs[:, -p:]  # Columns are the eigenvectors
        
    return proj_mat

# %% [markdown]
# ### Nearest neighbor classifier:

# %%
# Run through training dataset and create the mean embedding for all the images belonging to that class
def train_classifier(train_loader, lda_proj_mat, dim_reducer, p, num_classes=30):
    class_embs = [[] for _ in range(num_classes)]
    
    for batch in train_loader:
        img_vecn = batch["img_vecn"].numpy()
        labels = batch["y"] # Shape is (B,)
        
        if img_vecn.shape[0] >= p:
            
            if dim_reducer == "PCA":
                embs = get_PCA_vecs(img_vecn, p)
                
            elif dim_reducer == "LDA":
                embs = img_vecn @ lda_proj_mat  # Shape is (B, p)
                
            else:
                raise ValueError("Wrong input type: dim_reducer should be PCA or LDA")
            
            # Train Classifier embeddings
            for label in np.unique(labels):
                # idx of class_means = label + 1
                class_embedding = embs[labels == label]
                
                for sample in class_embedding:
                    class_embs[label - 1].append(sample)
                        
    return np.array(class_embs).astype(np.float32)

# %%
def run_testing_script(test_loader, lda_proj_mat, class_embs, dim_reducer, num_classes=30, num_samples=21):
    predicted_label = []
    true_label = []
    test_embs_list = [[] for _ in range(num_classes)]
    
    for batch in test_loader:
        img_vecn = batch["img_vecn"].numpy()
        labels = batch["y"] # Shape is (B,)
        
        if img_vecn.shape[0] >= p:
            if dim_reducer == "PCA":
                embs = get_PCA_vecs(img_vecn, p)
                
            elif dim_reducer == "LDA":
                # img_vecn = img_vec / np.linalg.norm(img_vec, axis=1, keepdims=True) # Shapes is (B, CHW)
                # The projection matrix is just the top eigenvectors?
                embs = img_vecn @ lda_proj_mat  # Shape is (B, p)
                
            else:
                raise ValueError("Wrong input type: dim_reducer should be PCA or LDA")
            # Compare nearest embeddings to get predicted label
            for embedding, label in zip(embs, labels):
                embedding = np.array(np.expand_dims(embedding, axis=0), dtype=np.float32)
                index = get_id_of_nearest_embedding(class_embs, embedding, num_classes, num_samples)

                test_embs_list[index.item()].append(embedding)
                predicted_label.append(index.item() + 1)
                true_label.append(label)
                
    return np.array(predicted_label, dtype=np.float32), np.array(true_label, dtype=np.float32), test_embs_list

# %%
accuracy_list_lda = []

# %%
# batch_size = 630
# num_classes = 30
# dim_reducer = "LDA"
# lda_option = "YUYANG"
# train_loader = DataLoader(dataset=DimReducerBuilder(TRAIN_DATA_PATH, option="bw"), batch_size=batch_size, shuffle=False)
# S_B, S_W = calculate_global_parameters_for_LDA(train_loader, num_classes=num_classes, feature_dim=4096)
# for p in range(3, 32):
#     if p % 24 == 0:
#         print(p)
#     lda_proj_mat = get_projection_matrix(S_B=S_B, S_W=S_W, p=p, lda_option=lda_option)
    
#     # Run through test set and find the nearest embedding and assign that label to the image
#     class_embs = train_classifier(train_loader, dim_reducer=dim_reducer, p=p, num_classes=num_classes, lda_proj_mat=lda_proj_mat)
    
#     test_loader = DataLoader(dataset=DimReducerBuilder(EVAL_DATA_PATH, option="bw"), batch_size=batch_size, shuffle=False)
#     pred_labels, true_labels, test_embs = run_testing_script(test_loader, lda_proj_mat, class_embs, dim_reducer=dim_reducer)
#     accuracy = np.count_nonzero(pred_labels == true_labels) / len(pred_labels)
#     accuracy_list_lda.append(np.round(accuracy * 100, 4))

# %%
p = 16 # [3, 8, 16]
batch_size = 630
num_classes = 30

# %%
dim_reducer = "PCA"
lda_option = "YUYANG" #"YUYANG"
train_loader = DataLoader(dataset=DimReducerBuilder(TRAIN_DATA_PATH, option="bw"), batch_size=batch_size, shuffle=False)


# %%
lda_proj_mat = None
if dim_reducer == "LDA":
    # Compute S_W and S_B:
    S_B, S_W = calculate_global_parameters_for_LDA(train_loader, num_classes=num_classes, feature_dim=4096)
    # Also decides whether we compute YUYANG or not:
    lda_proj_mat = get_projection_matrix(S_B=S_B, S_W=S_W, p=p, lda_option=lda_option)

# %%
# Run through test set and find the nearest embedding and assign that label to the image
class_embs = train_classifier(train_loader, dim_reducer=dim_reducer, p=p, num_classes=num_classes, lda_proj_mat=lda_proj_mat)

# %%
test_loader = DataLoader(dataset=DimReducerBuilder(EVAL_DATA_PATH, option="bw"), batch_size=batch_size, shuffle=False)
pred_labels, true_labels, test_embs = run_testing_script(test_loader, lda_proj_mat, class_embs, dim_reducer=dim_reducer)
accuracy = np.count_nonzero(pred_labels == true_labels) / len(pred_labels)
np.round(accuracy * 100, 4)

# %% [markdown]
# ### Graph the embeddings:

# %%
test_embs_to_print = []
for sample in test_embs:
    if len(sample) >= 10:
        test_embs_to_print.append(sample)

# %%
# Match the shapes of the train and test embeddings:
min_num_training_class_samples, min_num_test_class_samples = np.inf, np.inf
for train_sample, test_sample in zip(class_embs, test_embs_to_print):
    if len(train_sample) < min_num_training_class_samples:
        min_num_training_class_samples = len(train_sample)
    if len(test_sample) < min_num_test_class_samples:
        min_num_test_class_samples = len(test_sample)
        
print("Minimum training samples per class", min_num_training_class_samples)
print("Minimum testing samples per class", min_num_test_class_samples)        

graph_train_embs = np.zeros((num_classes, min_num_training_class_samples, p))
graph_test_embs = np.zeros((num_classes, min_num_test_class_samples, p))

for i, (train_sample, test_sample) in enumerate(zip(class_embs, test_embs_to_print)):
    test_sample = np.squeeze(test_sample)
    graph_train_embs[i] = np.array(train_sample[:min_num_training_class_samples], dtype=np.float32)
    graph_test_embs[i] = np.array(test_sample[:min_num_test_class_samples], dtype=np.float32)

# Reshape embeddings and generate labels
train_embeddings = graph_train_embs.reshape(-1, p)
train_labels = np.repeat(np.arange(num_classes), min_num_training_class_samples)

test_embeddings = graph_test_embs.reshape(-1, p)
test_labels = np.repeat(np.arange(num_classes), min_num_test_class_samples)

# Reduce to 2D with UMAP
umap_reducer = umap.UMAP(n_components=2)
train_umap = umap_reducer.fit_transform(train_embeddings)
test_umap = umap_reducer.transform(test_embeddings)


# %%
# Plot training data with different colors for each class
plt.figure(figsize=(8, 6))
scatter = plt.scatter(train_umap[:, 0], train_umap[:, 1], c=train_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, ticks=range(num_classes), label="Class Label")
plt.title("Training Data: UMAP Embeddings")
plt.axis("off")
plt.savefig(f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/PCA/UMAP/train_umap_{p}.jpg")
plt.close()

# Plot test data with predicted labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(test_umap[:, 0], test_umap[:, 1], c=test_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, ticks=range(num_classes), label="Ground Truth Label")
plt.title("Test Data: UMAP Embeddings with Predicted Labels")
plt.axis("off")
plt.savefig(f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/PCA/UMAP/test_umap_{p}.jpg")
plt.close()

# Generate and plot the confusion matrix:
cm = confusion_matrix(true_labels, pred_labels)

# Create a heatmap using Seaborn
plt.figure(figsize=(16, 16))
sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')

# Add labels and title
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.axis("off")
plt.savefig(f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/PCA/Confusion_Mat/conf_mat_{p}.jpg")
plt.close()

# %%
# Generate x-axis values (e.g., epoch numbers or iteration indices)
x_values = list(range(1, len(accuracy_list) + 1))
plt.plot(x_values, accuracy_list, marker='o', linestyle='-', label='PCA')
plt.plot(x_values, accuracy_list_lda, marker='x', linestyle='-', label='LDA')
plt.legend(loc="upper right")
plt.xlabel("Embedding Dim")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.savefig("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/Accuracy/PCA.jpg")


