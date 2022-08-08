# Modified from https://github.com/NikitaChizhov/deep_kalman_filter_for_BM
# Generates data for Healing MNIST dataset 

import mnist
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import os 
from os import path
import matplotlib.pyplot as plt 

def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img

def apply_noise(img, bit_flip_ratio):
    img = np.array(img)
    mask = np.random.random(size=(28,28)) < bit_flip_ratio
    img[mask] = 0
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
    rotation_steps = [360*2/seq_len]*seq_len
    for idx, rotation in enumerate(get_rotations(img, rotation_steps)):
        if idx >= squares_begin and idx < squares_end:
            rotation = apply_square(rotation, square_size)
        rotations.append(binarize(apply_noise(rotation, noise_ratio)))
    return rotations, rotation_steps

class HealingMNIST():
    def __init__(self, filepath, seq_len=5, square_count=3, square_size=5, noise_ratio=0.15, digits=range(10)):
        mnist_train = [(img, label) for img, label in zip(mnist.train_images()[:10000], mnist.train_labels()[:10000]) if label in digits]
        mnist_test = [(img, label) for img, label in zip(mnist.test_images()[:1000], mnist.test_labels()[:1000]) if label in digits]

        train_images = []
        test_images = []

        train_filepath = filepath + "train/"
        test_filepath = filepath + "test/"

        os.makedirs(train_filepath, exist_ok=True)
        os.makedirs(test_filepath, exist_ok=True)

        for i, (img, label) in tqdm(enumerate(mnist_train)):
            train_img, train_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio)
            train_img = np.asarray(train_img).astype(np.float32)
            image = np.zeros((seq_len, 32,32))
            image[:, 2:30, 2:30] = train_img
            np.savez(path.join(train_filepath, f"{i}"), images=image[...,None], label=label, rotations=train_rot)
        
        print(f"Successfully generated and saved train data in {train_filepath}.")

        for i, (img, label) in tqdm(enumerate(mnist_test)):
            test_img, test_rot = heal_image(img, seq_len, square_count, square_size, noise_ratio)
            test_img = np.asarray(test_img).astype(np.float32)
            image = np.zeros((seq_len, 32,32))
            image[:, 2:30, 2:30] = test_img
            np.savez(path.join(test_filepath, f"{i}"), images=image[...,None], label=label, rotations=train_rot)

        
if __name__ == "__main__": 
    # filepath = 'dataset/HealingMNIST/5/'
    # hmnist = HealingMNIST(filepath, seq_len=10, square_count=0, noise_ratio = 0)

    obj = np.load("dataset/HealingMNIST/5/train/2.npz")

    plt.imshow(np.concatenate(obj["images"], axis=1))
    plt.savefig("dataset/HealingMNIST/5/sample.jpeg")

    # obj = np.load("dataset/HealingMNIST/v2/train/1.npz")
    # plt.imshow(np.concatenate(obj["images"], axis=1))
    # plt.savefig("dataset/HealingMNIST/v2/sample1.jpeg")

    # obj = np.load("dataset/HealingMNIST/v2/train/2.npz")
    # plt.imshow(np.concatenate(obj["images"], axis=1))
    # plt.savefig("dataset/HealingMNIST/v2/samples/sample2.jpeg")
        