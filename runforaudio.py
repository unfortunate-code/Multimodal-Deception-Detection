import warnings
import os
import torch
import torchaudio
#import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.decomposition import PCA
from tqdm import tqdm 



import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F



import sys
import pdb
import time
import json
import logging
import pickle



logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


warnings.simplefilter('ignore')


from torchvggish import vggish, vggish_input


# Load an audio file
def load_audio_file(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate

def extract_audio_features(audio_file , vgg16):
    
    #audio_file = "/data/leticia/English/Control/cookie/002-0.mp3"
    embedding_model = vggish()
    embedding_model.eval()

    example = vggish_input.wavfile_to_examples(audio_file)
    with torch.no_grad():
        embeddings = embedding_model.forward(example)

    #print(embeddings.shape)
    features = embeddings

    return features

def pad_tensor(tensor, max_len, padding_value=0):
        pad_size = max_len - tensor.shape[0]
        #return torch.cat([tensor, torch.zeros(pad_size, tensor.shape[1])], dim=0)
        if tensor.shape[0] < max_len:
            pad_size = max_len - tensor.shape[0]
            #return #torch.cat([tensor, torch.zeros(pad_size, tensor.size(1))], dim=0)
            return torch.cat([torch.tensor(tensor[:max_len, :], dtype=torch.float32), torch.zeros(pad_size, tensor.shape[1])], dim=0)
        elif tensor.shape[0] > max_len:
            #return torch.cat([torch.tensor(tensor[:max_len, :], dtype=torch.float32), torch.zeros(pad_size, tensor.shape[1])], dim=0)
            return tensor[:max_len, :]   
        else:
            return tensor #torch.cat([torch.tensor(tensor, dtype=torch.float32), torch.zeros(pad_size, tensor.shape[1])], dim=0)

        
class AudioFeaturesDataset(Dataset):
    def __init__(self, file_paths, labels, vgg16, features_cache_dir='audiofeatures_cache2'):
        self.file_paths = file_paths
        self.labels = labels
        self.vgg16 = vgg16

        self.all_audio_features = []

        # Create the features_cache directory if it does not exist
        os.makedirs(features_cache_dir, exist_ok=True)
        print("Starting the feature extraction")
        for audio_file in file_paths:
            print("audio_file: ", audio_file)
            videoname_path = audio_file.split(".")
            cache_file = os.path.join(features_cache_dir, os.path.basename(videoname_path[0]) + '.pickle')
            cache_file_nopca = os.path.join('audiofeatureswithoutpca2', os.path.basename(videoname_path[0]) + '.pickle')


            #audio_features = extract_audio_features(self.file_paths[idx], self.vgg16)
            
            # Check if the cache file exists and load features if it does
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
            else:
                #features = extract_video_features(video_path)
                audio_features = extract_audio_features(audio_file, self.vgg16)
                with open(cache_file_nopca, 'wb') as f:
                    pickle.dump(audio_features, f)
                f.close()

                # Pad the tensor to a fixed length (replace `max_length` with the desired value)
                audio_features = pad_tensor(audio_features, max_len=50)
                # Apply PCA to reduce dimensionality
                pca = PCA(n_components=2)
                audio_features = pca.fit_transform(audio_features)

                # Save features to the cache file
                with open(cache_file, 'wb') as f:
                    pickle.dump(audio_features, f)
                f.close()
            self.all_audio_features.append(audio_features)

            #pca = PCA(n_components=2)
            #self.all_audio_features = pca.fit_transform(np.array(self.all_audio_features))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        audio_features = self.all_audio_features[idx]
        label = self.labels[idx]
        return torch.tensor(audio_features, dtype=torch.float32).squeeze(), torch.tensor(label, dtype=torch.long)


def create_dataset_and_loader(lies_folder, truth_folder, test_size=0.2, batch_size=10):
    lies_files  = [os.path.join(lies_folder, f) for f in os.listdir(lies_folder)]
    truth_files = [os.path.join(truth_folder, f) for f in os.listdir(truth_folder)]

    #test_folder = "/data/leticia/English/test_audio"
    #test_audio     = [os.path.join(test_folder, f) for f in os.listdir(test_folder)]

    file_paths = lies_files + truth_files
    labels = [0] * len(lies_files) + [1] * len(truth_files)
    #test2_labels = [0]*len(test_audio) 
    #print(labels) 
    #exit(0)


    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=test_size, stratify=labels)

    #train_dataset = AudioFeaturesDataset(train_files, train_labels)
    #test_dataset = AudioFeaturesDataset(test_files, test_labels)
    
    #vgg16 = load_vgg16()
    # Define VGG model
    vgg16 = models.vgg16(pretrained=True)
    train_dataset = AudioFeaturesDataset(train_files, train_labels, vgg16)
    test_dataset = AudioFeaturesDataset(test_files, test_labels, vgg16)
    #test2_dataset = AudioFeaturesDataset(test_audio, test2_labels, vgg16)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test2_loader = "" #### DataLoader(test2_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader, test2_loader

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 64)
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
    for inputs, labels in tqdm(train_loader):
        #print(f'inputs shape: {inputs.shape}')  # Add this line
        #print(f'labels shape: {labels.shape}')  # Add this line
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(f'outputs shape: {outputs.shape}')  # Add this line
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":

    #lies_folder = "/home/leticia/datasets/Audio/Deceptive"
    #truth_folder = "/home/leticia/datasets/Audio/Truthful"

    
    lies_folder = "/home/leticia/datasets/Dataset2/Audio/Lies"
    truth_folder = "/home/leticia/datasets/Dataset2/Audio/Truth"

    print("We are creating the dataset and dataloader")
    train_loader, test_loader, test2_loader = create_dataset_and_loader(lies_folder, truth_folder)
    print("We are done creating the dataset and dataloader")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #input_size = 50*10 #64000 #131072        #20000
    input_size = 10*10 #64000 #131072        #20000
    num_classes = 2
    
    print("Initializing our model")
    model = SimpleClassifier(input_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print("Time to start training our mdoel")
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
        #accuracy = evaluate(model, test2_loader, device)
        #print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Real audio Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "simple_classifier.pth")
