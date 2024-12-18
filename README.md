# Image Captioning with Deep Learning

This repository contains an implementation of an image captioning system using PyTorch and PyTorch Lightning. The model combines a CNN encoder (ResNet50) with an LSTM decoder to generate natural language descriptions of images.

## Project Overview

The system works by:
1. Encoding images using a pre-trained ResNet50 model
2. Decoding the image features using an LSTM network
3. Generating captions word by word using the decoder

## Features

- Uses PyTorch Lightning for organized training
- Pre-trained ResNet50 encoder
- LSTM decoder with attention
- NLTK integration for text preprocessing
- Supports the Flickr30k dataset
- Configurable model architecture and training parameters
- Dynamic teacher forcing during training


## Model Architecture

### Encoder
- ResNet50 pre-trained on ImageNet
- Removes final classification layer
- Projects features to embedding space

### Decoder
- LSTM-based decoder
- Word embedding layer
- Teacher forcing during training
- Greedy search for caption generation

## Training Details

- Learning rate: 1e-4
- Optimizer: Adam
- Learning rate scheduler: ReduceLROnPlateau
- Teacher forcing ratio: Configurable
- Optional encoder fine-tuning


## Acknowledgments

- ResNet architecture from torchvision
- PyTorch Lightning for training framework
- Flickr30k dataset
