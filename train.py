# Usage: python train.py data_directory
# Prints training loss, validation loss, and validation accuracy during training
# Options:
#     --save_dir save_directory: Set directory to save checkpoints
#     --arch "vgg13": Choose architecture
#     --learning_rate 0.01 --hidden_units 512 --epochs 20: Set hyperparameters
#     --gpu: Use GPU for training

# Sample bash command: python train.py './flowers' '../saved_models' --epochs 5

import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json

def data_transformation(args):
    # Define transformations, ImageFolder & DataLoader
    # Returns DataLoader objects for training and validation, then a class_to_idx dict
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # Validate paths before proceeding
    if not os.path.exists(args.data_directory):
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        raise FileNotFoundError
    if not os.path.exists(args.save_directory):
        print("Save Directory doesn't exist: {}".format(args.save_directory))
        raise FileNotFoundError
    # Additional validation for train and validation folders
    # ...

    # Define transformations for training and validation datasets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create ImageFolder and DataLoader objects for training and validation
    # ...

    return train_data_loader, valid_data_loader, train_data.class_to_idx

def train_model(args, train_data_loader, valid_data_loader, class_to_idx):
    # Train model, save model to directory, return True if successful
    # Build model using pre-trained VGG model and train it
    # ...

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='data_directory', help="Directory containing training images. Should have 'train' and 'valid' folders.")

    # Additional optional arguments
    # ...

    args = parser.parse_args()

    # Load and transform data
    train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)

    # Train and save model
    train_model(args, train_data_loader, valid_data_loader, class_to_idx)
