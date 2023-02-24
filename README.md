# Code Structure
1. Have the data in data folder as a subdirectory to your project folder
2. Have the embeddings folder as a subdirectory to your project folder
# Architecture
1. Extract features from each of the modalities
2. Pass the features through a fusion model
3. Pass the fused features through an MLP followed by a softmax output to detect truth or lie.
# Transcript Features
transcript_features_1 - BERT base

# Audio Features

# Video Features

# Affect features