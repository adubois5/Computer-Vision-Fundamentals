# %%
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
def apply_haar_filter(img_bw, haar_size):
    if haar_size % 2 == 1:
        # Odd -> add 1
        # Else this is already the largest even number > 4 sigma
        haar_size += 1
    haar_dx = np.vstack((-1*np.ones((haar_size, 1)), np.ones((haar_size, 1)) ))
    haar_dy = torch.tensor(-1*haar_dx.copy().T)
    haar_dx = torch.tensor(haar_dx)
    
    # haar_dx = torch.tensor([[-1] * haar_size + [1] * haar_size], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # haar_dy = -haar_dx.transpose(2, 3)

    # dx = cv2.filter2D(img_bw, -1, haar_dx)
    # dy = cv2.filter2D(img_bw, -1, haar_dy)
    
    dx = F.conv2d(img_bw, haar_dx, padding='same')
    dy = F.conv2d(img_bw, haar_dy, padding='same')

    return np.hstack((dx, dy))

def apply_haar_filter(img_bw, haar_size):
    haar_dx_np = np.vstack((-1 * np.ones((haar_size, 1)), np.ones((haar_size, 1))))
    haar_dy_np = -haar_dx_np.T

    # Convert Haar kernels to PyTorch tensors with proper shape
    haar_dx = torch.tensor(haar_dx_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    haar_dy = torch.tensor(haar_dy_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Ensure img_bw has the correct dimensions (batch_size, channels, height, width)
    if len(img_bw.shape) == 3:
        img_bw = img_bw.unsqueeze(0)  # Add batch dimension if missing

    # Apply Haar filters using F.conv2d
    dx = F.conv2d(img_bw, haar_dx, padding='same')
    dy = F.conv2d(img_bw, haar_dy, padding='same')
    return torch.hstack((dx, dy))

# %%
class DataBuilder(Dataset):
    def __init__(self, path, option=False):
        self.path = path
        self.image_list = [f for f in os.listdir(path) if f.endswith('.png')]
        self.len = len(self.image_list)
        self.aug = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fn = os.path.join(self.path, self.image_list[index])
        x = Image.open(fn).convert('RGB')
        x = self.aug(x)
        
        low_feature_vector = [torch.squeeze(torch.reshape(x, shape=(x.shape[0], -1)))]
        for haar_size in range(2, x.shape[1], 2):
            haar_img = apply_haar_filter(x, haar_size)
            pooled_img = F.avg_pool2d(haar_img, kernel_size=(haar_size, haar_size))
            pooled_img_flat = torch.squeeze(torch.reshape(pooled_img, shape=(pooled_img.shape[0], -1)))
            low_feature_vector.append(pooled_img_flat)
        feature_vecs = torch.hstack(low_feature_vector)
        
        return feature_vecs


# %%
pos_train_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/train/positive/"
neg_train_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/train/negative/"
pos_test_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/test/negative/"
neg_test_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/test/negative/"
num_pos_train = len(os.listdir(pos_train_path))
num_neg_train = len(os.listdir(neg_train_path))
num_pos_test = len(os.listdir(pos_test_path))
num_neg_test = len(os.listdir(neg_test_path))

# %%
pos_train_loader = DataLoader(dataset=DataBuilder(pos_train_path), batch_size=num_pos_train, shuffle=True)
neg_train_loader = DataLoader(dataset=DataBuilder(neg_train_path), batch_size=num_neg_train, shuffle=True)
pos_test_loader = DataLoader(dataset=DataBuilder(pos_test_path), batch_size=num_pos_test, shuffle=True)
neg_test_loader = DataLoader(dataset=DataBuilder(neg_test_path), batch_size=num_neg_test, shuffle=True)

# %%
for pos_train, neg_train, pos_test, neg_test in zip(pos_train_loader, neg_train_loader, pos_test_loader, neg_test_loader):
    print("Pos Train: ", pos_train.shape)
    print("Neg Train: ", neg_train.shape)
    # Create a matrix of all training images, positive or negative:
    train_imgs = np.vstack((pos_train, neg_train))
    train_labels = np.hstack(( np.ones(pos_train.shape[0]), -1*np.ones(neg_train.shape[0])) )
    
    # Randomize the order of the training images. Maintain a constant pairing of label to img though
    shuffle_indices = np.random.permutation(train_imgs.shape[0])
    train_imgs = train_imgs[shuffle_indices]
    train_labels = train_labels[shuffle_indices]
    
    # Do the same for thesting images:
    test_imgs = np.vstack((pos_test, neg_test))
    test_labels = np.hstack(( np.ones(pos_test.shape[0]), -1*np.ones(neg_test.shape[0])) )
    shuffle_indices = np.random.permutation(test_imgs.shape[0])
    test_imgs = test_imgs[shuffle_indices]
    test_labels = test_labels[shuffle_indices]
    
    print("Train imgs", train_imgs.shape)
    print("Test imgs", test_imgs.shape)

# %%
class WeakClassifier():
    def __init__(self):
        # These are the terms we need to calculate for the weak-classifier:
        self.best_feature = None
        self.best_threshold = None
        self.best_polarity = None
        self.min_error = np.inf

    def get_params(self, imgs, labels, weight_mat):
        # Normalize each weight matrix:
        weight_mat = weight_mat / np.sum(weight_mat)
    
        # We need to loop through each feature in the img matrix. Each feature is counted as a column in that matrix:
        for feature_idx in range(imgs.shape[1]):
            # Extract current feature (the column)
            features = imgs[:, feature_idx]
            
            # Sort the feature, weights, and labels
            sorted_indices = np.argsort(features)
            sorted_features = features[sorted_indices]
            sorted_weights = weight_mat[sorted_indices]
            sorted_labels = labels[sorted_indices]
            
            # We need to use multiplication here to preserve the original shape
            # However, we don't want the opposite labels to affect the cumulative summ
            S_plus = np.cumsum(sorted_weights * (sorted_labels == 1))
            S_minus = np.cumsum(sorted_weights * (sorted_labels == -1))
            T_plus = np.sum(sorted_weights * (sorted_labels == 1))
            T_minus = np.sum(sorted_weights * (sorted_labels == -1))
            
            # Calculate the polarity errors:
            e_1 = S_plus + T_minus - S_minus
            e_neg1 = S_minus + T_plus - S_plus
            
            # Calculate classification error:
            for i, feature in enumerate(sorted_features):
                # Since Error = min(e_1, e_neg1) we will always compute both and keep the trailing
                # minimum along both calculations
                if e_1[i] < self.min_error:
                    self.min_error = e_1[i]
                    self.best_feature = feature_idx
                    self.best_threshold = feature
                    self.best_polarity = 1
                if e_neg1[i] < self.min_error:
                    self.min_error = e_neg1[i]
                    self.best_feature = feature_idx
                    self.best_threshold = feature
                    self.best_polarity = -1
        return (self.best_feature, self.best_threshold, self.best_polarity, self.min_error)


# %%
class ClassifierCascade():
    def __init__(self):
        self.classifier_list = []
        self.alpha_list = []
        self.cascades = []
        self.max_num_classifiers_per_cascade = 5
        self.max_cascades = 3
        
    def run_strong_classifier(self, imgs, labels):
        # I associate a uniform initial weight with each image initially:
        weight_mat = np.ones(imgs.shape[0], dtype=np.float32) / imgs.shape[0]
        
        # Define the cascade:
        for classifier_idx in range(self.max_num_classifiers_per_cascade):
            # Every new iteration, we add in a new weak classifier until we have reached the maximum number, or they have the correct accuracy.
            self.classifier_list.append(WeakClassifier().get_params(imgs, labels, weight_mat))
            feature, threshold, polarity, error = self.classifier_list[classifier_idx]
            
            # Update algorithm parameters
            beta = error / (1 - error)
            alpha = np.log(1 / beta)
            self.alpha_list.append(alpha)
            
            print(f"Weak classifier id {(classifier_idx + 1):.3f} feature: {feature:.3f}, threshold: {threshold:.3f}, polarity: {polarity:.3f}, error: {error:.3f}, alpha: {alpha:.3f}")
            
            # Update weights accordingly:
            # You need to find your predictions (feature vs threshold feature value)
            # However, also make sure to take into account the polarity
            pred_labels = np.where((imgs[:, feature] * polarity) >= (threshold * polarity), 1, -1)
            wrong_preds = pred_labels != labels
            
            # Multiply by 1 where predictions are incorrect, else by beta.
            # Also normalize weights to make sure they sum to 1
            weight_mat = weight_mat * np.where(wrong_preds, 1, beta)
            weight_mat /= np.sum(weight_mat)
            
            # Calculate the accuracy of all weak classifiers so far:
            # To do so we first need to get the predictions by passing the input through all the weak classifiers
            # We can then get the sign of this predictions to assign it to a class label (1 or -1)
            predictions_so_far = np.zeros_like(labels)
            # Get the predictions for each classifier, alpha is a weight factor for how much each classifier contributes
            for (f, th, p, _), alpha in zip(self.classifier_list, self.alpha_list):
                predictions_so_far += alpha * np.where((imgs[:, f] * p) >= (th * p), 1, -1)
            final_pred_labels = np.sign(predictions_so_far)
            
            # Calculate accuracy
            accuracy = np.count_nonzero(final_pred_labels == labels) / len(final_pred_labels)
            print(f"Iteration {classifier_idx + 1}: Accuracy = ", np.round(accuracy, 3))
            
            # Evaluate if there are enough classifiers:
            if accuracy > 1.00:
                break
        return self.classifier_list, self.alpha_list

    def run_cascades(self, imgs, labels):
        # This function will run multiple cascades until the false positive rate reaches 0
        for cascade_idx in range(self.max_cascades):
            # We first train a strong classifier
            classifier_params, alphas = ClassifierCascade().run_strong_classifier(imgs, labels)
            self.cascades.append((classifier_params, alphas))
            
            # We then need to evaluate the most recent cascade as we did before
            predictions_so_far = np.zeros_like(labels)
            for (f, th, p, _), alpha in zip(classifier_params, alphas):
                predictions_so_far += alpha * np.where((imgs[:, f] * p) >= (th * p), 1, -1)
            cascade_pred_labels = np.sign(predictions_so_far)
            
            # false_positives = np.mean((cascade_pred_labels == 1) & (labels == -1))
            false_positives = np.count_nonzero((cascade_pred_labels == 1) & (labels == -1)) / len(cascade_pred_labels)
            false_negatives = np.count_nonzero((cascade_pred_labels == -1) & (labels == 1)) / len(cascade_pred_labels)
            accuracy = np.count_nonzero(cascade_pred_labels == labels) / len(cascade_pred_labels)
            print(f"Cascade id: {cascade_idx + 1}, False positive rate: {false_positives:.2f}, False negative rate: {false_negatives:.2f} Final Accuracy: {accuracy:.2f}")
            
            # Now that we know the false positive rate, we can either terminate, or keep going by removing correctly labeled negatives:
            if false_positives + false_negatives < 0.01:
                break
            # Remove all the correctly classified negative images from the dataset
            idx_to_keep = (labels == 1) | ((cascade_pred_labels == 1) & (labels == -1))
            imgs = imgs[idx_to_keep]
            labels = labels[idx_to_keep]
            
            # We also need to stop if there are no more imgs left (not removing any)
            if len(idx_to_keep) == len((labels == 1)):
                break
            
        return self.cascades
    
    def test_cascade(self, imgs, labels):
        total_num_images = imgs.shape[0]
        final_true_negatives = np.zeros_like(labels)
        final_true_positives = np.zeros_like(labels)
        
        # Get the accuracy:
        for cascade_idx, (classifier_params, alphas) in enumerate(self.cascades):
            # Get the predictions on the test dataset:
            predictions_so_far = np.zeros_like(labels)
            for (f, th, p, _), alpha in zip(classifier_params, alphas):
                predictions_so_far += alpha * np.where((imgs[:, f] * p) >= (th * p), 1, -1)
            cascade_pred_labels = np.sign(predictions_so_far)

            # Compute performance metrics for the current cascade
            true_negative_mask = (cascade_pred_labels == -1) & (labels == -1)
            true_positive_mask = (cascade_pred_labels == 1) & (labels == 1)
            final_true_negatives = np.logical_or(final_true_negatives, true_negative_mask)
            final_true_positives = np.logical_or(final_true_positives, true_positive_mask)
            
            false_positives = np.count_nonzero((cascade_pred_labels == 1) & (labels == -1))
            false_negatives = np.count_nonzero((cascade_pred_labels == -1) & (labels == 1))
            true_negatives = np.count_nonzero(true_negative_mask)
            true_positives = np.count_nonzero(true_positive_mask)
            tot_negative = np.count_nonzero(labels == -1)
            tot_positive = np.count_nonzero(labels == 1)
            
            print(f"True Positives: {true_positives}, True Negatives: {true_negatives}, Total Image Count {total_num_images}")
            false_positive_rate = false_positives / (tot_negative)
            false_negative_rate = false_negatives / (tot_positive)
            accuracy = (true_positives + true_negatives) / len(cascade_pred_labels)
            percent_of_dataset_kept = len(cascade_pred_labels) / total_num_images
            print(f"Cascade id: {cascade_idx + 1}, False positive rate: {false_positive_rate:.2f}, False negative rate: {false_negative_rate:.2f}, Accuracy: {np.round(accuracy*100, 4)}, over {np.round(percent_of_dataset_kept*100, 4)} of the imgs.")

        # Print final accuracy:
        final_tp = np.count_nonzero(final_true_positives)
        final_tn = np.count_nonzero(final_true_negatives)
        final_accuracy = (final_tp + final_tn) / total_num_images
        print(f"Final Accuracy: {np.round(final_accuracy*100, 4)}")
        

# %%
classifier_cascade = ClassifierCascade()
cascade_params = classifier_cascade.run_cascades(train_imgs.copy(), train_labels.copy())

# %%
classifier_cascade.test_cascade(test_imgs.copy(), test_labels.copy())

# %%



