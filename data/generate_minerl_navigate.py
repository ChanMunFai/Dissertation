import tensorflow as tf
import tensorflow_datasets as tfds
import minerl_navigate
import os
from os import path
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

filepath = "dataset/MinecraftRL/"
train_filepath = filepath + "train/"
os.makedirs(train_filepath, exist_ok=True)

dataset = tfds.load('minerl_navigate', shuffle_files=False)
train = dataset['train']
train = train.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(
        tf.reshape(x['video'], (1, 500, 64, 64, 3))))
train = train.batch(1)

counter = 0 
for i, batch in enumerate(tqdm(train)): 
    batch = batch.numpy()
    print(batch.shape)
    chunks = np.split(batch, 5, axis = 1)
    
    for vid in chunks: 
        counter += 1 
        vid = np.squeeze(vid)
        np.savez(path.join(train_filepath, f"{counter}"), vid) 
     

### Test 
# test_filepath = filepath + "test/"
# os.makedirs(test_filepath, exist_ok=True)

# dataset = tfds.load('minerl_navigate', shuffle_files=False)
# test = dataset['test']
# test = test.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(
#         tf.reshape(x['video'], (1, 500, 64, 64, 3))))
# test = test.batch(1)

# for i, batch in enumerate(tqdm(test)): 
#     batch = batch.numpy()
#     np.savez(path.join(test_filepath, f"{i}"), batch) 

# obj = np.load("dataset/MinecraftRL/train/0.npz")
# obj = obj["arr_0"]
# sample_img = obj[0,:20,:,:,:]
# # sample_img = np.transpose(sample_img, (0, 3, 1, 2))
# print(sample_img.shape)
# sample_img = np.concatenate(sample_img, axis=0)
# print(sample_img.shape)

# plt.imshow(sample_img)
# plt.savefig("dataset/MinecraftRL/train/sample1.jpeg")
