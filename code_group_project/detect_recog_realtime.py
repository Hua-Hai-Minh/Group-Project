import torch
import torchvision.transforms as transforms
import cv2
from model.resnet import Resnet34Triplet
from model.face_detect_yunet.yunet import YuNet
from PIL import Image
import time

flag_gpu_available = torch.cuda.is_available()
if flag_gpu_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint = torch.load('model/model_resnet34_triplet.pt', map_location=device)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
best_distance_threshold = checkpoint['best_distance_threshold']

model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(size=140),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6071, 0.4609, 0.3944],
        std=[0.2457, 0.2175, 0.2129]
    )
])

# Load YuNet for face detection
yynet = YuNet(modelPath='./face_detect_yunet/face_detection_yunet_2023mar.onnx',
              inputSize=[320, 320],
              confThreshold=0.9,
              nmsThreshold=0.3,
              topK=5000,
              backendId=cv2.dnn.DNN_BACKEND_OPENCV,
              targetId=cv2.dnn.DNN_TARGET_CPU)

# Open the camera (you may need to adjust the index based on your camera setup)
cap = cv2.VideoCapture(0)

# Initialize FPS calculation
fps_start_time = time.time()
fps_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform face detection using YuNet
    yynet.setInputSize([frame.shape[1], frame.shape[0]])
    faces = yynet.infer(frame)

    # Check if any faces are detected
    if faces.shape[0] > 0:
        # Take the first detected face (you may modify this logic based on your requirements)
        detected_face = faces[0][:-1]

        # Preprocess the detected face for ResNet34
        detected_face_pil = Image.fromarray(detected_face)

        # Apply the updated preprocessing directly to PIL Image
        detected_face_tensor = preprocess(detected_face_pil)
        detected_face_tensor = detected_face_tensor.unsqueeze(0)
        detected_face_tensor = detected_face_tensor.to(device)

        # Perform face recognition using ResNet34
        embedding = model(detected_face_tensor)

        # Convert the embedding Torch Tensor to a Numpy array
        embedding = embedding.cpu().detach().numpy()

        # Extract bounding box coordinates
        x, y, w, h = faces[0][:4]  # Adjust the index to match the structure of the faces array

        # Draw bounding box on the original image
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the face vector to a text file
        with open("face_vector.txt", "w") as file:
            for value in embedding.flatten():
                file.write(str(value) + "\n")

        print("Face vector saved to face_vector.txt")
    else:
        print("No faces detected.")

    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_counter / (time.time() - fps_start_time)
        print(f"FPS: {fps:.2f}")
        fps_counter = 0
        fps_start_time = time.time()

    # Display the resulting frame with FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

