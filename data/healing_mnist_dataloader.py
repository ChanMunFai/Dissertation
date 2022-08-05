import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from matplotlib import pyplot as plt, animation
import cv2
from glob import glob
from tqdm import tqdm 

class HealingMNISTSmall(Dataset):

    def __init__(self,path,seen_len=None):
        self.seen_len = seen_len # manually specify split for seen vs unseen data
        self.data = np.load(path)
        self.data = np.array(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        im = self.data[i]
        im = im[:,np.newaxis,:,:]
        im = 1 * im 

        if self.seen_len == None: 
            seq_len = len(im)//2
        else: 
            seq_len = self.seen_len 

        seq, target = im[:seq_len], im[seq_len:]
        
        return seq, target

def visualize_rollout(rollout, interval=50, show_step=False, save=False):
    """Visualization for a single sample rollout of a physical system.
    Args:
        rollout (numpy.ndarray): Numpy array containing the sequence of images. It's shape must be
            (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    fig = plt.figure()
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape))
            cv2.putText(
                black_img, str(i), (0, 30), fontScale=0.22, color=(255, 255, 255), thickness=1,
                fontFace=cv2.LINE_AA)
            res_img = (im + black_img / 255.) / 2
        else:
            res_img = im
        img.append([plt.imshow(res_img, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=10000)
    if save:
        writergif = animation.PillowWriter(fps=2)
        ani.save('dataset/HealingMNIST/v1/healing_sequence.gif', writergif)
    plt.show()

def create_datasets(filepath, train, seen_len):
    if train == True: 
        filepath = filepath + "train/"
    else: 
        filepath = filepath + "test/"

    filelist = os.listdir(filepath)
    datasets = []

    print("Loading dataset")
    for i in tqdm(filelist):
        datasets.append(HealingMNISTSmall(f"{filepath}{i}", seen_len))

    return datasets 

class HealingMNISTDataLoader(ConcatDataset): 
    def __init__(self,path,train=True,seen_len=None):
        self.path = path 
        self.train = train 
        self.seen_len = seen_len
        self.datasets = create_datasets(self.path, self.train, self.seen_len)

        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        
        self.cumulative_sizes = self.cumsum(self.datasets)

if __name__ == '__main__':
    # ds = HealingMNISTSmall(path = 'dataset/HealingMNIST/v1/train/train_1.npy')
    # loader = torch.utils.data.DataLoader(
    #     ds,
    #     shuffle=False,
    #     num_workers=0,
    #     batch_size=2
    # )
    # data, target = next(iter(loader))

    dataset = HealingMNISTDataLoader(path = 'dataset/HealingMNIST/v1/', train = True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        batch_size=2
    )

    data, target = next(iter(train_loader))
    print(data.size(), target.size())
    print(torch.min(data), torch.max(data))

    ### Manual way of loading dataset 
    # datasets = create_datasets(filepath = 'dataset/HealingMNIST/v1/', train = True, seen_len = None)
    # dataset = ConcatDataset(datasets)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     shuffle=False,
    #     num_workers=0,
    #     batch_size=2
    # )
    
    # data_np = data[1].detach().numpy()
    # data_np = data_np.transpose((0, 2, 3, 1))
    # print(data_np.shape)
    # visualize_rollout(data_np, interval=5, show_step=False, save=True)

    