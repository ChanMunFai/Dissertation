# Modified from https://github.com/NikitaChizhov/deep_kalman_filter_for_BM
# Generates data for Healing MNIST dataset 

import mnist
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import os 

def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img
    
def apply_noise(img, bit_flip_ratio):
    img = np.array(img)
    mask = np.random.random(size=(28,28)) < bit_flip_ratio
    img[mask] = 255 - img[mask]
    return img

def get_rotations(img, rotation_steps):
    for rot in rotation_steps:
        img = scipy.ndimage.rotate(img, rot, reshape=False)
        yield img

def binarize(img):
    return img > 127

def heal_image(img, seq_len, square_count, square_size, noise_ratio):
    squares_begin = np.random.randint(0, seq_len - square_count)
    squares_end = squares_begin + square_count

    rotations = []
    rotation_steps = np.random.random(size=seq_len) * 180 - 90

    for idx, rotation in enumerate(get_rotations(img, rotation_steps)):
        if idx >= squares_begin and idx < squares_end:
            rotation = apply_square(rotation, square_size)
        rotations.append(binarize(apply_noise(rotation, noise_ratio)))

    return rotations, rotation_steps

class HealingMNIST():
    def __init__(self, filepath, seq_len=5, square_count=3, square_size=5, noise_ratio=0.15, digits=range(10)):
        mnist_train = [img for img, label in zip(mnist.train_images(), mnist.train_labels()) if label in digits]
        mnist_test = [img for img, label in zip(mnist.test_images(), mnist.test_labels()) if label in digits]

        train_images = []
        test_images = []

        train_filepath = filepath + "train/"
        test_filepath = filepath + "test/"

        if not os.path.isdir(train_filepath):
            os.makedirs(train_filepath)

        if not os.path.isdir(test_filepath):
            os.makedirs(test_filepath)

        file_num = 1
        for img in tqdm(mnist_train):
            train_img, train_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio)
            train_images.append(train_img)

            if len(train_images) == 5000: 
                train_images = np.array(train_images)
                np.save(f'{train_filepath}train_{file_num}.npy', train_images) 
                print(f"Saved images in {train_filepath}train_{file_num}.npy")
                file_num += 1 
                train_images = []
        
        file_num = 1 
        for img in tqdm(mnist_test):
            test_img, test_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio)
            test_images.append(test_img)

            if len(test_images) == 5000: 
                test_images = np.array(test_images)
                np.save(f'{test_filepath}test_{file_num}.npy', test_images) 
                print(f"Saved images in {test_filepath}test_{file_num}.npy")
                file_num += 1 
                test_images = []
            
        print("Successfully generated and saved data.")

if __name__ == "__main__": 
    filepath = 'dataset/HealingMNIST/v1/'
    hmnist = HealingMNIST(filepath, seq_len=40, square_count=0, noise_ratio = 0)

        