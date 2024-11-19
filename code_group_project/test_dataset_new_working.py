import torch
import torchvision.transforms as transforms
import cv2
from model.resnet import Resnet34Triplet
from model.face_detect_yunet.yunet import YuNet
from model.face_detect_yunet.alignment import alignment_procedure
from PIL import Image
import joblib
import os
import numpy as np
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

# Define data directory
data_dir = '/home/medievalvn/Desktop/AgeDB_Database/archive/AgeDB'

# Get a list of image files in the dataset folder
image_files = [file for file in os.listdir(data_dir) if file.endswith(".jpg")]

# Extract labels and paths from image files
labels = []
image_paths = []

for file in image_files:
    parts = file.split('_')
    label = parts[1]  # Assuming the name is the second part of the filename
    img_path = os.path.join(data_dir, file)

    labels.append(label)
    image_paths.append(img_path)

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
for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Processing Images"):
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
model_filename = 'svm_model_custom_dataset.joblib'
joblib.dump(clf, model_filename)
print(f"Trained SVM model saved to {model_filename}")

# Load the trained SVM model for testing
loaded_clf = joblib.load(model_filename)

# Make predictions on the validation set with tqdm for a progress bar
with tqdm(total=X_val.shape[0], desc="Testing SVM on Validation Set") as pbar:
    y_val_pred = loaded_clf.predict(X_val)
    pbar.update(X_val.shape[0])

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