#Importing key dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Constants
DATA_DIR = './data'
download_dataset = False

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

# Load MNIST training and test datasets with transformations
train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

# Define data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batch_size, shuffle=False)



# Define neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer: 28*28=784, Output layer: 128
        self.fc2 = nn.Linear(128, 64)      # Hidden layer: 128, Output layer: 64
        self.fc3 = nn.Linear(64, 10)       # Hidden layer: 64, Output layer: 10 (number of classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input images
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)          # Output layer, no activation as it's included in the loss function
        return x

# Initialize the neural network
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross entropy loss for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001




# Training loop
epochs = 7  # Number of epochs
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader.dataset)}")
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader.dataset)}\n")

print("Training finished!")

# Save the trained model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved as 'mnist_model.pth'")

# Evaluation on test set
true_labels = []
predicted_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print(f"Precision on test set: {precision:.2f}")
print(f"Recall on test set: {recall:.2f}")
print(f"F1-score on test set: {f1:.2f}")
with open('log.txt', 'a') as f:
    f.write(f"Accuracy on test set: {accuracy * 100:.2f}%\n")
    f.write(f"Precision on test set: {precision:.2f}%\n")
    f.write(f"Recall on test set: {recall:.2f}%\n")
    f.write(f"F1-score on test set: {f1:.2f}%\n")


# Define function to classify an image
def classify_image(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Pass the image through the model
    with torch.no_grad():
        output = model(image)
    
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Interactive classification loop
while True:
    with open('log.txt', 'a') as f:
        f.write(f"Please enter a filepath (or 'exit' to quit): %\n")
    filepath = input("Please enter a filepath (or 'exit' to quit): ")
    if filepath.lower() == 'exit':
        with open('log.txt', 'a') as f:
            f.write(f"Exiting... %\n")
        print("Exiting...")
        break
    if not os.path.exists(filepath):
        with open('log.txt', 'a') as f:
            f.write(f"File not found. Please enter a valid filepath. %\n")
        print("File not found. Please enter a valid filepath.")
        continue
    try:
        predicted_class = classify_image(filepath)
        print("Classifier:", predicted_class)
        with open('log.txt', 'a') as f:
            f.write(f"Classifier: {predicted_class} %\n")
    except Exception as e:
        with open('log.txt', 'a') as f:
            f.write(f"Error: {e} %\n")
        print("Error:", e)
