import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from src.models.autoencoder import Autoencoder
from src.detectors.ocsvm_detector import OCSVMDetector
import numpy as np

# Configuration
ENCODED_SPACE_DIM = 8
FC2_INPUT_DIM = 32 * 3 * 3 # Based on the CNN output size
BATCH_SIZE = 64
EPOCHS = 10

def train_autoencoder(model, dataloader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        for batch_features, _ in dataloader:
            # Reshape image to be 1 channel
            batch_features = batch_features.view(-1, 1, 28, 28)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
        print(f"Autoencoder Epoch {epoch+1}, Loss: {loss.item():.4f}")

def train_ocsvm(autoencoder_model, dataloader):
    # Extract features using the trained autoencoder encoder
    features = []
    for batch_features, _ in dataloader:
        batch_features = batch_features.view(-1, 1, 28, 28)
        latent = autoencoder_model.encoder(batch_features).detach().numpy()
        features.append(latent)
    X_normal = np.concatenate(features)

    ocsvm = OCSVMDetector()
    ocsvm.train(X_normal)
    print("One-Class SVM trained.")
    return ocsvm

def main():
    # Load MNIST dataset as an example of normal data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize and train Autoencoder
    autoencoder = Autoencoder(ENCODED_SPACE_DIM, FC2_INPUT_DIM)
    train_autoencoder(autoencoder, train_dataloader)

    # Train One-Class SVM on latent features
    ocsvm_detector = train_ocsvm(autoencoder, train_dataloader)

    # Example of anomaly detection (using a subset of test data)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_features = []
    for batch_features, _ in test_dataloader:
        batch_features = batch_features.view(-1, 1, 28, 28)
        latent = autoencoder.encoder(batch_features).detach().numpy()
        test_features.append(latent)
    X_test = np.concatenate(test_features)

    anomalies_indices = ocsvm_detector.detect_anomalies(X_test)
    print(f"Detected {len(anomalies_indices)} anomalies in the test set.")

if __name__ == "__main__":
    main()
