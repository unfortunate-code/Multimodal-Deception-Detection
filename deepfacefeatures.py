import cv2
from deepface import DeepFace

# Set the path for the video
#video_path = '/home/leticia/datasets/Real-life_Deception_Detection_2016/Clips/Deceptive/trial_lie_001.mp4' #'video.mp4'

#it's a 128 array of numbers
# Process the video using a specific model
#model_name = 'OpenFace'
#process_video(video_path, model_name)

import os
import pickle
import cv2
from deepface import DeepFace

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video

import torch.nn as nn
from torchvision.models.video import r3d_18
import torch.nn.functional as F
from torchvision.transforms import Resize, Normalize, Compose

from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import pickle
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
import dlib 

class DeceptionDataset(Dataset):
    def __init__(self, file_paths, labels, model_name, frame_interval=30, transform=None, features_cache_dir='deepfacefeatures_cache'):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.model_name = model_name
        self.frame_interval = frame_interval
        features_cache_dir = model_name + "features_cache2"
        

        if model_name == 'Dlib':
            face_recognition_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
            self.dlib_face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)
            self.dlib_face_detector = dlib.get_frontal_face_detector()
            self.dlib_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


        # Create the features_cache directory if it does not exist
        os.makedirs(features_cache_dir, exist_ok=True)
        
        print("Started the features extraction")
        self.all_features = []
        for video_path in self.file_paths:
            # Create a cache file path for the current video
            videoname_path = video_path.split(".")
            cache_file = os.path.join(features_cache_dir, os.path.basename(videoname_path[0]) + '.pickle')
            
            # Check if the cache file exists and load features if it does
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
            else:
                features = self.process_video(video_path)
                # Save features to the cache file
                with open(cache_file, 'wb') as f:
                    pickle.dump(features, f)
            
            self.all_features.append(features)

        print("Finished the features extraction")
    def __getitem__(self, idx):
        return self.all_features[idx].to(torch.float), self.labels[idx]

    def __len__(self):
        return len(self.labels)
    
    def get_face_featuresold(self, image):
        features = DeepFace.represent(img_path=image, model_name=self.model_name, enforce_detection=False)
        return features
    
    def get_face_features(self, image):
        if self.model_name == 'Dlib':
            # Detect face
            dets = self.dlib_face_detector(image, 1)
            if len(dets) == 0:
                return None
            else:
                face = dets[0]

            # Get landmarks
            shape = self.dlib_shape_predictor(image, face)

            # Compute face descriptor (128D)
            face_descriptor = self.dlib_face_recognition_model.compute_face_descriptor(image, shape)
            return np.array(face_descriptor)
        else:
            features = DeepFace.represent(img_path=image, model_name=self.model_name, enforce_detection=False)
            return features

    def process_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        features_list = []

        while cap.isOpened():
            ret, frame = cap.read()

            # Break the loop if we reach the end of the video
            if not ret:
                break

            # Process every frame_interval-th frame
            if frame_count % self.frame_interval == 0:
                try:
                    # Extract face features
                    face_features = self.get_face_features(frame)
                    face_features_array = np.array(face_features[0]["embedding"])
                    features_list.append(face_features_array)
                except Exception as e:
                    print(f"Frame {frame_count}, Error: {e}")

            frame_count += 1

        # Release the video capture object
        cap.release()

        # Pad or truncate the features list to a fixed length (e.g., 16)
        
        emb_length = len(face_features[0]["embedding"])
        print(emb_length)
        fixed_length = 16
        if len(features_list) < fixed_length:
            padding = [np.zeros(emb_length) for _ in range(fixed_length - len(features_list))]
            features_list.extend(padding)
        else:
            features_list = features_list[:fixed_length]

        # Convert the list of features to a tensor and return
        return torch.tensor(features_list)



class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        #print('forward x.shape: ', x.shape)
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=0.15)
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, p=0.15)
        x = self.fc3(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        print("Inputs size:", inputs.size())
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc

def create_dataset_and_loader(deception_folder, truthful_folder, test_size=0.2, batch_size=10, model_name = 'OpenFace'):
    deception_files  = [os.path.join(deception_folder, f) for f in os.listdir(deception_folder)]
    truthful_files = [os.path.join(truthful_folder, f) for f in os.listdir(truthful_folder)]


    file_paths = deception_files + truthful_files
    labels = [0] * len(deception_files) + [1] * len(truthful_files)
    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=test_size, stratify=labels)

    train_dataset = DeceptionDataset(train_files, train_labels, model_name)
    test_dataset = DeceptionDataset(test_files, test_labels, model_name)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #print(train_loader.shape, test_loader.shape)

    return train_loader, test_loader

if __name__ == '__main__':
    

    # Set up data transforms
    transform = Compose([
                          Resize((256, 256)),  # resize frames to 256x256
                          transforms.ToTensor(),
                          Normalize(mean=[0.5, 0.5, 0.5], 
                                    std=[0.5, 0.5, 0.5])  # normalize pixel values
                        ])


    # Set up training and validation data loaders

    print("We are creating the dataset and dataloader")
    #deception_folder = "/home/leticia/datasets/Real-life_Deception_Detection_2016/Clips/Deceptive"
    #truthful_folder = "/home/leticia/datasets/Real-life_Deception_Detection_2016/Clips/Truthful"

    ##second dataset:
    deception_folder = "/home/leticia/datasets/Dataset2/Videos/Lie/"
    truthful_folder = "/home/leticia/datasets/Dataset2/Videos/Truth/"

    #models_total = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'Dlib', 'ArcFace']
    models_total = ['OpenFace','VGG-Face', 'Facenet', 'DeepID', 'Dlib', 'ArcFace']
    models_emb_size = [128, 2622, 128, 160, 150, 512]


    for index in range(0, len(models_total)):

        print("We are done creating the dataset and dataloader for ", models_total[index], " model ")
        train_loader, val_loader = create_dataset_and_loader(deception_folder, truthful_folder, model_name = models_total[index])
        
        # Set up the model, loss function, and optimizer
        #model = SimpleClassifier(input_size=16*128, num_classes=2) OpenFace
        model = SimpleClassifier(input_size=16*int(models_emb_size[index]), num_classes=2)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Train and evaluate the model for a fixed number of epochs
        num_epochs = 10
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
