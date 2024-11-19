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
import time  # Import the time module

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet model
checkpoint = torch.load('model/model_resnet34_triplet.pt', map_location=device)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define data directory
data_dir = '/home/medievalvn/Documents/USTH_project/group_project/facenet-pytorch-glint360k-master/user_images_processed'

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
labels = []

# Record start time
start_time = time.time()

# Loop through each person's folder
for person_name in os.listdir(data_dir):
    person_folder = os.path.join(data_dir, person_name)
    
    # Loop through each image in the person's folder
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        
        # Preprocess the image for ResNet34
        img_pil = Image.open(img_path)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        
        # Extract features using ResNet34
        with torch.no_grad():
            embedding = model(img_tensor)
        
        # Convert the embedding Torch Tensor to a Numpy array
        embedding = embedding.cpu().detach().numpy().flatten()
        
        # Print information for debugging
        #print(f"Person: {person_name}, Embedding: {embedding}")

        # Append features and labels
        features.append(embedding)
        labels.append(person_name)


# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)
# Calculate and print the total runtime
total_runtime = time.time() - start_time
print(f"Total Runtime: {total_runtime:.2f} seconds")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
#clf = svm.NuSVC(nu=0.01, kernel='linear', probability=True)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Save the trained SVM model to a file
model_filename = 'svm_model.joblib'
joblib.dump(clf, model_filename)
print(f"Trained SVM model saved to {model_filename}")

# Load the trained SVM model for testing
loaded_clf = joblib.load(model_filename)

# Make predictions on the test set
y_pred = loaded_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Print information for debugging
print(f"Training Labels: {y_train}")
print(f"Testing Labels: {y_test}")

# Print decision scores for debugging
decision_scores = loaded_clf.decision_function(X_test)
print(f"Decision Scores: {decision_scores}")

