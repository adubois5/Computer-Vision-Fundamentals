# %%
import os
import numpy as np
import torch
from torch import nn, optim
import umap
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ### Accuracies:
# 
# p = 3: 0%  
# 

# %%
class DataBuilder(Dataset):
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
        
        return {'x': x, 'y': self.label_list[index]}


# %%
def get_id_of_nearest_embedding(training_set, probe_embedding):
    # We first calculate the euclidean distance between the probe and all trained embeddings:
    distances = np.linalg.norm(training_set - probe_embedding, axis=1)

    # Then we return the index with the smallest distance
    return np.argmin(distances)

# %%
def train(epoch, vae_loss, model, optimizer, trainloader):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad()
        input = data["x"].to(device)
        mu, logvar = model.encode(input)
        z = model.reparameterize(mu, logvar)
        xhat = model.decode(z)
        loss = vae_loss(xhat, input, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(trainloader.dataset)))
    return model
class VaeLoss(nn.Module):
    def __init__(self):
        super(VaeLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, xhat, x, mu, logvar):
        loss_MSE = self.mse_loss(xhat, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD

# %%
class Autoencoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, encoded_space_dim * 2)
        )
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 4 * 4 * 64),
            nn.LeakyReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu, logvar = x[:, :self.encoded_space_dim], x[:, self.encoded_space_dim:]
        return mu, logvar

    def decode(self, z):
        x = self.decoder_lin(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


# %% [markdown]
# ### VAE Training:

# %%
##################################
# Change these
p = 8
batch_size = 24
training = False
TRAIN_DATA_PATH = '/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/train/'
EVAL_DATA_PATH = '/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/test/'
LOAD_PATH = f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/exp/model_{p}.pt"
OUT_PATH = '/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/exp/'
##################################

# %%
train_loader = DataLoader(dataset=DataBuilder(TRAIN_DATA_PATH, option="bw"), batch_size=batch_size, shuffle=True)


# %%
accuracy_list = []

# %%
p = 16
LOAD_PATH = f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/exp/model_{p}.pt"
train_loader = DataLoader(dataset=DataBuilder(TRAIN_DATA_PATH, option="bw"), batch_size=batch_size, shuffle=True)
training=False
model = Autoencoder(p).to(device)

if training:
    epochs = 100
    log_interval = 1
    trainloader = DataLoader(
        dataset=DataBuilder(TRAIN_DATA_PATH, option=""),
        batch_size=32,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    vae_loss = VaeLoss()
    for epoch in range(1, epochs + 1):
        model = train(epoch, vae_loss, model, optimizer, trainloader)
    torch.save(model.state_dict(), os.path.join(OUT_PATH, f'model_{p}.pt'))
else:
    trainloader = DataLoader(
        dataset=DataBuilder(TRAIN_DATA_PATH, option=""),
        batch_size=1,
    )
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()

    X_train, y_train = [], []
    for batch_idx, data in enumerate(trainloader):
        mu, logvar = model.encode(data['x'].to(device))
        z = mu.detach().cpu().numpy().flatten()
        X_train.append(z)
        y_train.append(data['y'].item())
    X_train = np.stack(X_train)
    y_train = np.array(y_train)

    testloader = DataLoader(
        dataset=DataBuilder(EVAL_DATA_PATH, option=""),
        batch_size=1,
    )
    X_test, y_test = [], []
    for batch_idx, data in enumerate(testloader):
        mu, logvar = model.encode(data['x'].to(device))
        z = mu.detach().cpu().numpy().flatten()
        X_test.append(z)
        y_test.append(data['y'].item())
    X_test = np.stack(X_test)
    y_test = np.array(y_test)


# %%
train_embs = [[] for _ in range(30)]
test_embs = [[] for _ in range(30)]

for i, (train_emb, train_label, test_emb, test_label) in enumerate(zip(X_train, y_train, X_test, y_test)):
    train_embs[train_label - 1].append(train_emb)
    test_embs[test_label - 1].append(test_emb)


# %%
train_embs = np.array(train_embs, dtype=np.float32)
test_embs = np.array(test_embs, dtype=np.float32)

# %%
# Now that all the array is reordered, I can add them to the faiis search
num_classes, num_samples, embedding_dim = train_embs.shape
flattened_train_embs = train_embs.reshape(-1, embedding_dim)
print("Flattened", flattened_train_embs.shape)

# Create a mapping from flattened indices to class IDs
index_to_class_id = np.repeat(np.arange(num_classes), num_samples)
print("indices", index_to_class_id.shape)



# %%
true_labels = []
pred_labels = []
for test_emb, test_label in zip(X_test, y_test):
    search_emb = np.array(np.expand_dims(test_emb, axis=0), dtype=np.float32)
    index = get_id_of_nearest_embedding(flattened_train_embs, search_emb)
    true_labels.append(test_label)
    pred_labels.append(index_to_class_id[index.item()] + 1)
true_labels = np.array(true_labels)
pre_labels = np.array(pred_labels)
accuracy = np.count_nonzero(pred_labels == true_labels) / len(pred_labels)
print("Accuracy: ", np.round(accuracy * 100, 4), "\n")
accuracy_list.append(np.round(accuracy * 100, 4))

# %%
# Generate x-axis values (e.g., epoch numbers or iteration indices)
x_values = list(range(1, len(accuracy_list) + 1))
plt.plot(x_values, accuracy_list, marker='o', linestyle='-', label='Accuracy')
plt.grid(True)

plt.savefig("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/Accuracy/Autoencoder.jpg")

# %%
num_classes, num_samples, embedding_dim = train_embs.shape
flattened_train_embs = train_embs.reshape(-1, embedding_dim)
train_labels = np.repeat(np.arange(num_classes), num_samples)

num_classes, num_samples, embedding_dim = test_embs.shape
flattened_test_embs = test_embs.reshape(-1, embedding_dim)
test_labels = np.repeat(np.arange(num_classes), num_samples)

# %%
# Reduce to 2D with UMAP
umap_reducer = umap.UMAP(n_components=2)
train_umap = umap_reducer.fit_transform(flattened_train_embs)
test_umap = umap_reducer.transform(flattened_test_embs)

# %%
# Plot training data with different colors for each class
plt.figure(figsize=(8, 6))
scatter = plt.scatter(train_umap[:, 0], train_umap[:, 1], c=train_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, ticks=range(num_classes), label="Class Label")
plt.title("Training Data: UMAP Embeddings")
plt.axis("off")
plt.savefig(f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/AutoEncoder/UMAP/train_umap_{p}.jpg")
plt.close()

# Plot test data with predicted labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(test_umap[:, 0], test_umap[:, 1], c=test_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, ticks=range(num_classes), label="Ground Truth Label")
plt.title("Test Data: UMAP Embeddings with Predicted Labels")
plt.axis("off")
plt.savefig(f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/AutoEncoder/UMAP/test_umap_{p}.jpg")
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
plt.savefig(f"/mnt/cloudNAS3/Adubois/Classes/ECE661/HW10/Results/AutoEncoder/Confusion_Mat/conf_mat_{p}.jpg")
plt.close()

# %%



