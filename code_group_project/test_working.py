import torch
import torchvision.transforms as transforms
from model.resnet import Resnet34Triplet  # Assuming you have the ResNet model implementation
from sklearn import svm
import joblib
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet model
checkpoint = torch.load('model/model_resnet34_triplet.pt', map_location=device)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define data directory and annotation file
data_dir = '/home/medievalvn/Desktop/celeba_test2/img_align_celeba'
annotation_file = '/home/medievalvn/Downloads/identity_CelebA.txt'

# Read annotation file to get image paths and labels
with open(annotation_file, 'r') as file:
    lines = file.readlines()

image_paths = []
labels = []

# Keep track of the number of identities and images
identity_counter = 0
image_counter = 0

# Dictionary to store image paths for each identity
identity_images = {}

for line in lines:
    parts = line.split()
    img_path = os.path.join(data_dir, parts[0])
    label = int(parts[1])

    # Check if we have already seen this identity
    if label not in identity_images:
        identity_images[label] = [img_path]
        identity_counter += 1
    else:
        # Limit the number of images per identity to 20
        if len(identity_images[label]) < 30:
            identity_images[label].append(img_path)
            image_counter += 1

    # Break if we reach the desired number of identities and images
    if identity_counter >= 10 and image_counter >= 300:
        break

# Flatten the dictionary to get the final image_paths and labels
for label, images in identity_images.items():
    image_paths.extend(images)
    labels.extend([label] * len(images))

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(size=140),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6071, 0.4609, 0.3944],
        std=[0.2457, 0.2175, 0.2129]
    )
])

# Initialize lists to store features and labels
features = []

# Loop through each image in the dataset with tqdm for a progress bar
for idx, (img_path, label) in enumerate(tqdm(zip(image_paths, labels), total=len(image_paths), desc="Processing Images")):
    # Preprocess the image for ResNet34
    img_pil = Image.open(img_path)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # Extract features using ResNet34
    with torch.no_grad():
        embedding = model(img_tensor)

    # Convert the embedding Torch Tensor to a Numpy array
    embedding = embedding.cpu().detach().numpy().flatten()

    # Append features
    features.append(embedding)

# Convert list to numpy array
X = np.array(features)
y = np.array(labels)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train the SVM classifier with tqdm for a progress bar
with tqdm(total=X_train.shape[0], desc="Training SVM") as pbar:
    clf = svm.SVC(kernel='linear', C=1.0, probability=True)
    clf.fit(X_train, y_train)
    pbar.update(X_train.shape[0])

# Save the trained SVM model to a file
model_filename = 'svm_model_celeba_subset.joblib'
joblib.dump(clf, model_filename)
print(f"Trained SVM model saved to {model_filename}")

# Load the trained SVM model for testing
loaded_clf = joblib.load(model_filename)

# Make predictions on the validation set with tqdm for a progress bar
with tqdm(total=X_val.shape[0], desc="Testing SVM on Validation Set") as pbar:
    y_val_pred = loaded_clf.predict(X_val)
    pbar.update(X_val.shape[0])

# Make predictions on the validation set
y_val_pred = loaded_clf.predict(X_val)

# Print predicted labels for investigation
print(f"Predicted Labels on Validation Set: {y_val_pred}")

# Calculate accuracy, precision, recall, and F1 on validation set
accuracy_val = accuracy_score(y_val, y_val_pred)

# Set average='micro' to avoid warnings and zero divisions
precision_val = precision_score(y_val, y_val_pred, average='micro')
recall_val = recall_score(y_val, y_val_pred, average='micro')
f1_val = f1_score(y_val, y_val_pred, average='micro')

print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
print(f"Validation Precision: {precision_val:.2f}")
print(f"Validation Recall: {recall_val:.2f}")
print(f"Validation F1 Score: {f1_val:.2f}")

# Make predictions on the test set with tqdm for a progress bar
with tqdm(total=X_test.shape[0], desc="Testing SVM on Test Set") as pbar:
    y_test_pred = loaded_clf.predict(X_test)
    pbar.update(X_test.shape[0])

# Calculate accuracy, precision, recall, and F1 on test set
accuracy_test = accuracy_score(y_test, y_test_pred)

# Set average='micro' to avoid warnings and zero divisions
precision_test = precision_score(y_test, y_test_pred, average='micro')
recall_test = recall_score(y_test, y_test_pred, average='micro')
f1_test = f1_score(y_test, y_test_pred, average='micro')

print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
print(f"Test Precision: {precision_test:.2f}")
print(f"Test Recall: {recall_test:.2f}")
print(f"Test F1 Score: {f1_test:.2f}")