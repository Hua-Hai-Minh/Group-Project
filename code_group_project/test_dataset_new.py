import torch
import torchvision.transforms as transforms
import cv2
from model.resnet import Resnet34Triplet
from model.face_detect_yunet.yunet import YuNet
from model.face_detect_yunet.alignment import alignment_procedure
from PIL import Image
import numpy as np
import joblib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time  # Import the time module
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet model
checkpoint = torch.load('model/model_resnet34_triplet.pt', map_location=device)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load YuNet for face detection
yynet = YuNet(modelPath='./model/face_detect_yunet/face_detection_yunet_2023mar.onnx',
              inputSize=[320, 320],
              confThreshold=0.9,
              nmsThreshold=0.3,
              topK=5000,
              backendId=cv2.dnn.DNN_BACKEND_OPENCV,
              targetId=cv2.dnn.DNN_TARGET_CPU)

# Define preprocessing steps, including normalization
mean = [0.6071, 0.4609, 0.3944]
std = [0.2457, 0.2175, 0.2129]

preprocess = transforms.Compose([
    transforms.Resize(size=140),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Set the threshold for face recognition confidence
threshold = 0.1
# Initialize max_input_width
max_input_width = 640
# Define data directory
data_dir = '/home/medievalvn/Desktop/AgeDB_Database/archive/AgeDB'

# Initialize lists to store features and labels
features = []
labels = []
# Record start time
start_time = time.time()
# Loop through each image in the dataset with tqdm for a progress bar
for file in tqdm(os.listdir(data_dir), desc="Processing Images"):
    if file.endswith(".jpg"):
        # Load the image
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path)

        # Resize image if width is over max_input_width
        if img.shape[1] > max_input_width:
            scale_factor = max_input_width / img.shape[1]
            new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
            img = cv2.resize(img, new_size)

        # Perform face detection using YuNet
        yynet.setInputSize([img.shape[1], img.shape[0]])
        faces = yynet.infer(img)

        # Initialize variables outside the loop
        x, y, w, h = 0, 0, 0, 0

        # Perform face recognition for each detected face
        if faces.shape[0] > 0:
            for i in range(faces.shape[0]):
                detected_face = faces[i][:-1]
                # Perform face alignment using the alignment procedure
                bbox = faces[i][:4].astype(np.int32)

                # Ensure bounding box coordinates are within valid range
                x, y, w, h = bbox
                x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

                left_eye = tuple(faces[i][4:6].astype(np.int32))
                right_eye = tuple(faces[i][6:8].astype(np.int32))
                detected_face_aligned = alignment_procedure(img, left_eye, right_eye, (x, y, w, h))

                # Preprocess the aligned face for ResNet34
                detected_face_aligned_pil = Image.fromarray(detected_face_aligned)
                detected_face_aligned_tensor = preprocess(detected_face_aligned_pil)
                detected_face_aligned_tensor = detected_face_aligned_tensor.unsqueeze(0)
                detected_face_aligned_tensor = detected_face_aligned_tensor.to(device)

                # Extract features using ResNet34
                with torch.no_grad():
                    embedding = model(detected_face_aligned_tensor)

                # Convert the embedding Torch Tensor to a Numpy array
                embedding = embedding.cpu().detach().numpy().flatten()

                # Append features and labels
                features.append(embedding)
                labels.append(file.split('_')[1])  # Assuming the label is in the filename

# Convert list to numpy array
X = np.array(features)
y = np.array(labels)

if np.any(pd.isna(y)):
    raise ValueError("Labels contain NaN values. Please check your data.")

total_runtime = time.time() - start_time
print(f"Total Runtime: {total_runtime:.2f} seconds")

# Convert string labels to numerical labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train the SVM classifier
clf = svm.NuSVC(nu=0.01, kernel='linear', probability=True)
#clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = clf.predict(X_val)

# Evaluation metrics for validation set
accuracy_val = accuracy_score(y_val, y_val_pred)
precision_val = precision_score(y_val, y_val_pred, average='weighted')
recall_val = recall_score(y_val, y_val_pred, average='weighted')
f1_val = f1_score(y_val, y_val_pred, average='weighted')

print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
print(f"Validation Precision: {precision_val * 100:.2f}%")
print(f"Validation Recall: {recall_val * 100:.2f}%")
print(f"Validation F1 Score: {f1_val * 100:.2f}%")

# Make predictions on the test set
y_test_pred = clf.predict(X_test)

# Evaluation metrics for test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')

print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
print(f"Test Precision: {precision_test * 100:.2f}%")
print(f"Test Recall: {recall_test * 100:.2f}%")
print(f"Test F1 Score: {f1_test * 100:.2f}%")
