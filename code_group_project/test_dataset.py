import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from CelebADataset import CelebADataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from tqdm import tqdm
from models.resnet import Resnet34Triplet
from losses.triplet_loss import TripletLoss
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser(description="Testing a pre-trained FaceNet facial recognition model using Triplet Loss.")
parser.add_argument('--model_path', '-m', type=str, required=True,
                    help="Path to the pre-trained model checkpoint file"
                    )
parser.add_argument('--celeba_dataroot', type=str, required=True,
                    help="Absolute path to the CelebA dataset folder"
                    )
parser.add_argument('--celeba_annotation_file', type=str, required=True,
                    help="Path to the CelebA annotation file"
                    )
parser.add_argument('--embedding_dimension', default=512, type=int,
                    help="Dimension of the embedding vector (default: 512)"
                    )
parser.add_argument('--batch_size', default=1000, type=int,
                    help="Batch size (default: 544)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 4)"
                    )
parser.add_argument('--image_size', default=140, type=int,
                    help="Input image size (default: 140)"
                    )
args = parser.parse_args()

# Define image data pre-processing transforms
data_transforms = transforms.Compose([
    transforms.Resize(size=args.image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6071, 0.4609, 0.3944],
        std=[0.2457, 0.2175, 0.2129]
    )
])

def set_model_architecture(embedding_dimension, pretrained_model_path):
    model = Resnet34Triplet(
        embedding_dimension=embedding_dimension,
        pretrained=False  # Since you are loading a pre-trained model, set pretrained to False
    )
    model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
    print(f"Using InceptionResnetV2 model with embedding dimension {embedding_dimension}.")
    return model

def set_model_gpu_mode(model):
    flag_test_gpu = torch.cuda.is_available()
    if flag_test_gpu:
        model.cuda()
        print('Using GPU for testing.')
    else:
        print('Using CPU for testing.')

    return model

def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings = model(imgs)
    return embeddings

def main():
    model_path = args.model_path
    celeba_dataroot = args.celeba_dataroot
    celeba_annotation_file = args.celeba_annotation_file
    embedding_dimension = args.embedding_dimension
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size

    # Instantiate model
    model = set_model_architecture(
        embedding_dimension=embedding_dimension,
        pretrained_model_path=model_path
    )

    # Load model to GPU if available
    model = set_model_gpu_mode(model)

    # Load CelebA dataset
    celeba_dataset = CelebADataset(
        root_dir=celeba_dataroot,
        annotation_file=celeba_annotation_file,
        transform=data_transforms
    )

    celeba_dataloader = torch.utils.data.DataLoader(
        dataset=celeba_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    print("Total number of batches:", len(celeba_dataloader))

    # Evaluate on CelebA dataset
    model.eval()
    ground_truth_labels = []

    with torch.no_grad():
        embeddings_list = []

        print("Testing on CelebA dataset! ...")
        progress_bar = enumerate(tqdm(celeba_dataloader))

        for batch_index, (data, labels) in progress_bar:
            data = data.cuda()
            embeddings = model(data)
            embeddings_list.append(embeddings.cpu().detach().numpy())
            ground_truth_labels.extend(labels.cpu().numpy())

        # Flatten the list of embeddings
        embeddings_array = np.concatenate(embeddings_list, axis=0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(embeddings_array, ground_truth_labels, test_size=0.2, random_state=42)

        # Initialize and train the SVM classifier
        clf = svm.SVC(kernel='linear', C=1.0, probability=True)
        clf.fit(X_train, y_train)

        # Save the trained SVM model to a file
        model_filename = 'svm_model.joblib'
        joblib.dump(clf, model_filename)
        print(f"Trained SVM model saved to {model_filename}")

        # Load the trained SVM model for testing
        loaded_clf = joblib.load(model_filename)

        # Make predictions on the test set
        decision_scores = loaded_clf.decision_function(X_test)

        # Apply a threshold to decision scores to obtain binary predictions
        threshold = 0.0  # You may need to adjust this based on your data
        predictions = (decision_scores > threshold).astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Print information for debugging
        print(f"Testing Labels: {y_test}")
        print(f"Predictions: {predictions}")
        # Calculate and print additional metrics like precision and ROC AUC
        precision = precision_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        print(f"Precision: {precision:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == '__main__':
    main()