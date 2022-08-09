import mnist
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import os 
from os import path
import matplotlib.pyplot as plt 
from cv2 import cv2

class DancingMNIST():
    def __init__(self, filepath, seq_len=5, digits=range(10), train_len = 10000, test_len = 1000):
        self.mnist_train = [(img, label) for img, label in zip(mnist.train_images()[:train_len], mnist.train_labels()[:train_len]) if label in digits]
        self.mnist_test = [(img, label) for img, label in zip(mnist.test_images()[:test_len], mnist.test_labels()[:test_len]) if label in digits]        
        
        self.filepath = filepath 
        self.basic_cycle = ["down", "up", "up", "down"]
        self.seq_len = seq_len 

    def generate_train(self): 
        train_filepath = self.filepath + "train/"
        os.makedirs(train_filepath, exist_ok=True)

        for i, (img, label) in tqdm(enumerate(self.mnist_train)): 
            train_img = self.transform_image(img)
            train_img = np.asarray(train_img).astype(np.float32)
            train_img = np.where(train_img > 127, 0.5, 0) # binarise
            np.savez(path.join(train_filepath, f"{i}"), images=train_img)
        
        print(f"Successfully generated and saved train data in {train_filepath}.")

    def generate_test(self): 
        test_filepath = self.filepath + "test/"
        os.makedirs(test_filepath, exist_ok=True)

        for i, (img, label) in tqdm(enumerate(self.mnist_test)): 
            test_img = self.transform_image(img)
            test_img = np.asarray(test_img).astype(np.float32)
            test_img = np.where(test_img > 127, 0.5, 0) # binarise
            np.savez(path.join(test_filepath, f"{i}"), images=test_img)
        
        print(f"Successfully generated and saved test data in {test_filepath}.")

        
    def create_shift_sequence(self, seq_len): 
        shift_seq = []

        basic_cycle = self.shift_seq 
        shift_seq.extend(basic_cycle * int(seq_len//len(basic_cycle)))
        shift_seq.extend(basic_cycle[:(seq_len%len(basic_cycle))])
        
        return shift_seq
        
    def transform_image(self, patch): 
        rotation_steps = [360*2/self.seq_len]*self.seq_len
        shift_steps = self.create_shift_sequence(self.seq_len) 
        transformed_images = []

        position = "centre"

        for angle, shift_action in zip(rotation_steps, shift_steps): 
            patch = self.rotate(patch, angle)
            position = self.get_position(position, shift_action)
            img = self.shift_by_padding(patch, position)
            transformed_images.append(img)

        return transformed_images

    def rotate(self, img, angle):
        """Rotate a 28 X 28 patch"""

        rotated = scipy.ndimage.rotate(img, angle, reshape=False)

        return rotated 

    def shift_by_padding(self, patch, position): 
        """ Shifts a 28 X 28 patch by padding it accordingly. 
        Arguments: 
            patch: 28 X 28 
        Returns: 
            img: 32 X 32
        """
        img = np.zeros((32, 32))

        if position == "centre": 
            img[2:30, 2:30] = patch 
        elif position == "up": 
            img[0:28, 2:30] = patch 
        elif position == "down": 
            img[4:32, 2:30] = patch 
        elif position == "left": 
            img[2:30, 0:28] = patch 
        elif position == "right": 
            img[2:30, 4:32] = patch 

        return img 

    def get_position(self, position, action): 
        
        if position == "centre": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up"
            elif action == "down": 
                position = "down"
            elif action == "left": 
                position = "left"
            elif action == "right": 
                position = "right"
            return position 

        elif position == "up": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up"
            elif action == "down": 
                position = "centre"
            elif action == "left": 
                position = "up_left"
            elif action == "right": 
                position = "up_right"
            return position 

        elif position == "down": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "centre"
            elif action == "down": 
                position = "down"
            elif action == "left": 
                position = "down_left"
            elif action == "right": 
                position = "down_right"
            return position 

        elif position == "left": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up_left"
            elif action == "down": 
                position = "down_left"
            elif action == "left": 
                position = "left"
            elif action == "right": 
                position = "centre"
            return position 

        elif position == "right": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up_right"
            elif action == "down": 
                position = "down_right"
            elif action == "left": 
                position = "centre"
            elif action == "right": 
                position = "right"
            return position 

        elif position == "up_right": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up_right"
            elif action == "down": 
                position = "right"
            elif action == "left": 
                position = "up"
            elif action == "right": 
                position = "up_right"
            return position 

        elif position == "up_left": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up_left"
            elif action == "down": 
                position = "left"
            elif action == "left": 
                position = "up_left"
            elif action == "right": 
                position = "up"
            return position 

        elif position == "up_right": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "up_right"
            elif action == "down": 
                position = "right"
            elif action == "left": 
                position = "up"
            elif action == "right": 
                position = "up_right"
            return position 

        elif position == "down_right": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "right"
            elif action == "down": 
                position = "down_right"
            elif action == "left": 
                position = "down"
            elif action == "right": 
                position = "down_right"
            return position 

        elif position == "down_left": 
            if action == 0: 
                return position 
            elif action == "up": 
                position = "left"
            elif action == "down": 
                position = "down_left"
            elif action == "left": 
                position = "down_left"
            elif action == "right": 
                position = "down"
            return position 

if __name__ == "__main__": 
    filepath = 'dataset/DancingMNIST/20/'
    dancing_mnist = DancingMNIST(filepath, seq_len=5, train_len = 1, test_len = 1)

    obj = np.load("dataset/DancingMNIST/20/train/0.npz")

    # plt.imshow(np.concatenate(obj["images"], axis=1))
    # plt.savefig("dataset/HealingMNIST/5/sample.jpeg")       
