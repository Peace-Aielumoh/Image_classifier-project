# Image Classification Project

## Overview
This project implements a deep learning model to classify images into various categories. It uses a pre-trained model for feature extraction and training on a custom dataset of images.

## Features
- Utilizes transfer learning with pre-trained models from torchvision.
- Allows training on custom datasets for specific image classification tasks.
- Provides functionality for predicting classes of new images using the trained model.

## Dependencies
- Python (3.x)
- PyTorch
- torchvision
- NumPy
- PIL

## Getting Started
1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Prepare your dataset and organize it into appropriate folders (train, validation, test).
4. Train the model by running `python train.py data_directory`.
5. Predict classes of new images using `python predict.py /path/to/image checkpoint`.

## Usage
### Training
