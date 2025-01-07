import struct
from array import array
import random as rand
from neuronet import NeuralNetwork 
import numpy as np

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    
    return images

def load_labels(file_path):
    labels = []
    with open(file_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read()) 
    
    labels = np.array(labels)
    return labels


# Load the images
images = load_images("mnist_digits/t10k-images.idx3-ubyte")
labels = load_labels("mnist_digits/t10k-labels.idx1-ubyte")
 
      
def main():
    nn = NeuralNetwork(images, labels, [28*28, 200, 10], 255.0)
    nn.train()   


if __name__ == "__main__":
    main()

