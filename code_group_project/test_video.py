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

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet model
checkpoint = torch.load('model/model_resnet34_triplet.pt', map_location=device)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load the trained SVM model
model_filename = 'svm_model.joblib'
loaded_clf = joblib.load(model_filename)

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

# Initialize max_input_width
max_input_width = 720

# Set the threshold for face recognition confidence
threshold = 0.4

# Open the camera (you may need to adjust the index based on your camera setup)
cap = cv2.VideoCapture('/home/medievalvn/Desktop/demo/demo/demo/1.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables outside the loop
fps = 0  # Initialize fps outside the loop
smallest_detected_face_width = float('inf')  # Initialize with infinity
smallest_detected_face_height = float('inf')  # Initialize with infinity
smallest_recognized_face_width = float('inf')  # Initialize with infinity
smallest_recognized_face_height = float('inf')  # Initialize with infinity

# Initialize tick meter
tm = cv2.TickMeter()

# Initialize best_class_probabilities before the loop
best_class_probabilities = 0.0
best_name = " "

# Define output directories for recognized and unknown faces
recognized_faces_dir = '/home/medievalvn/Documents/USTH_project/group_project/facenet-pytorch-glint360k-master/recognized_faces'
unknown_faces_dir = '/home/medievalvn/Documents/USTH_project/group_project/facenet-pytorch-glint360k-master/unknown_faces'

# Create output directories if they don't exist
os.makedirs(recognized_faces_dir, exist_ok=True)
os.makedirs(unknown_faces_dir, exist_ok=True)

# Initialize variables for face saving
face_count = 0
unknown_face_count = 0

# Set the desired width and height for the displayed frame
display_width = 1280
display_height = 720

while True:
    # Start the tick meter for the current frame
    tm.reset()
    tm.start()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if frame is None:
        print("Error: Could not read frame from the video.")
        break

    # Resize frame if width is over max_input_width
    if frame.shape[1] > max_input_width:
        scale_factor = max_input_width / frame.shape[1]
        new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
        frame = cv2.resize(frame, new_size)

    # Perform face detection using YuNet
    yynet.setInputSize([frame.shape[1], frame.shape[0]])
    faces = yynet.infer(frame)

    # Initialize variables outside the loop
    x, y, w, h = 0, 0, 0, 0

    # Perform face recognition for each detected face
    if faces.shape[0] > 0:
        print(f"Number of faces detected: {faces.shape[0]}")  # Print the number of faces
        for i in range(faces.shape[0]):
            # Take each detected face
            detected_face = faces[i][:-1]

            # Perform face alignment using the alignment procedure
            bbox = faces[i][:4].astype(np.int32)

            # Ensure bounding box coordinates are within valid range
            x, y, w, h = bbox
            x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

            # Update the smallest_detected_face_width and smallest_detected_face_height if needed
            if w < smallest_detected_face_width:
                smallest_detected_face_width = w
            if h < smallest_detected_face_height:
                smallest_detected_face_height = h

            left_eye = tuple(faces[i][4:6].astype(np.int32))
            right_eye = tuple(faces[i][6:8].astype(np.int32))
            detected_face_aligned = alignment_procedure(frame, left_eye, right_eye, (x, y, w, h))

            # Preprocess the aligned face for ResNet34
            detected_face_aligned_pil = Image.fromarray(detected_face_aligned)
            detected_face_aligned_tensor = preprocess(detected_face_aligned_pil)
            detected_face_aligned_tensor = detected_face_aligned_tensor.unsqueeze(0)
            detected_face_aligned_tensor = detected_face_aligned_tensor.to(device)

            # Perform face recognition using ResNet34
            embedding = model(detected_face_aligned_tensor)

            # Convert the embedding Torch Tensor to a Numpy array
            embedding = embedding.cpu().detach().numpy().flatten()

            # Make predictions using the loaded SVM model
            predicted_probabilities = loaded_clf.predict_proba([embedding])

            # Extract the probability scores for each class
            class_probabilities = predicted_probabilities[0]

            # Find the index of the class with the highest probability
            best_class_index = np.argmax(class_probabilities)

            # Get the best class probability
            best_class_probabilities = class_probabilities[best_class_index]

            # Get the best class name
            best_name = loaded_clf.classes_[best_class_index]

            # Print information for debugging
            print(f"Predicted label: {best_name}, Probability: {best_class_probabilities}")

            # Check if the confidence is above the threshold
            if best_class_probabilities >= threshold:
                match_info = f"Predicted label: {best_name}, Confidence: {best_class_probabilities:.2f}"

                # Save recognized faces in folders based on the recognized person's name
                recognized_person_folder = os.path.join(recognized_faces_dir, best_name)
                os.makedirs(recognized_person_folder, exist_ok=True)
                face_count += 1
                face_filename = os.path.join(recognized_person_folder, f'recognized_face_{face_count}.jpg')
                cv2.imwrite(face_filename, detected_face_aligned)

                # Update the smallest_recognized_face_width and smallest_recognized_face_height if needed
                if w < smallest_recognized_face_width:
                    smallest_recognized_face_width = w
                if h < smallest_recognized_face_height:
                    smallest_recognized_face_height = h
            else:
                match_info = "Unknown Face"

                # Save unknown faces
                unknown_face_count += 1
                unknown_face_filename = os.path.join(unknown_faces_dir, f'unknown_face_{unknown_face_count}.jpg')
                cv2.imwrite(unknown_face_filename, detected_face_aligned)

            # Draw bounding box on the original image with white color and smaller size
            x, y, w, h = faces[i][:4]  # Adjust the index to match the structure of the faces array
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # Display the match information with red color and thinner text
            cv2.putText(frame, match_info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Stop the tick meter for the current frame
    tm.stop()

    # Get the frames per second for the current frame
    fps = tm.getFPS()

    # Display FPS
    print(f"FPS: {fps:.2f}")

    # Resize the frame before displaying
    frame = cv2.resize(frame, (display_width, display_height))

    # Display the resulting frame with FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the smallest width and height for a detected face and a recognized face
print(f"Smallest width for a detected face: {smallest_detected_face_width}")
print(f"Smallest height for a detected face: {smallest_detected_face_height}")
print(f"Smallest width for a recognized face: {smallest_recognized_face_width}")
print(f"Smallest height for a recognized face: {smallest_recognized_face_height}")

# Release the camera, close all windows, and release the video writer
cap.release()
cv2.destroyAllWindows()
