# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 00:17:12 2023

@author: raj18011
"""
import os
import torch
import torchvision
import torchvision.datasets as datasets
#mnist_trainset = datasets.MNIST(root='I:/Mnist/data', train=False, download=True, transform=None)
import numpy as np

from tqdm import tqdm
from PIL import Image

def save_mnist_as_images(images, labels, output_dir):
    for i, (image, label) in tqdm(enumerate(zip(images, labels))):
        output_directory = os.path.join(output_dir, str(label))
        if not os.path.exists(output_directory):
            
            os.makedirs(output_directory)
            
            
        img = Image.fromarray(image)
        img.save(f"{output_directory}/mnist_{i}.png")

# Replace these file paths with the paths to your MNIST .ubyte files
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Replace these file paths with the paths to your MNIST .ubyte
images_file_path = 'I:/Mnist/data/MNIST/raw/t10k-images-idx3-ubyte'
labels_file_path = 'I:/Mnist/data/MNIST/raw/t10k-labels-idx1-ubyte'
# Read images and labels using the previously defined function read_idx()
images = read_idx(images_file_path)
labels = read_idx(labels_file_path)

# Specify the output directory where you want to save the images
output_directory = 'I:/Mnist/data/MNIST/Processed'




# Check if the directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
else:
    print(f"Directory '{output_directory}' already exists.")


# Save MNIST images as PNG files in the specified output directory
save_mnist_as_images(images, labels, output_directory)
