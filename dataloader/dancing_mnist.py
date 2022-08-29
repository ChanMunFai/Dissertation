import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob

class DancingMNISTDataLoader(Dataset):
    def __init__(self,root_dir, train, seen_len = None):
        self.train = train 
        if self.train == True: 
            self.sequence_dir = root_dir + "train/"
        elif self.train == False: 
            self.sequence_dir = root_dir + "test/"

        try:
            self.sequence_filenames = glob(f"{self.sequence_dir}/*")
        except:
            raise ValueError("dir incorrect")

        self.seen_len = seen_len 

    def __len__(self):
        return len(self.sequence_filenames)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        sequence_path = self.sequence_filenames[idx]
        stored_obj = np.load(sequence_path)
        im = stored_obj["images"]
        im = im[:, np.newaxis]

        if self.seen_len == None: 
            seq_len = len(im)//2
        else: 
            seq_len = self.seen_len 
        seq, target = im[:seq_len], im[seq_len:]

        return seq, target

if __name__ == "__main__": 
    train_dataset = DancingMNISTDataLoader("dataset/DancingMNIST/20/v1/", train = True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=0,
        batch_size=2
    )
    # print(len(train_dataset))
    data, target = next(iter(train_loader))
    print(data.shape, target.shape)
    # print(torch.max(data), torch.min(data))

    # val_dataset = DancingMNISTDataLoader("dataset/DancingMNIST/20/", train = False)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     shuffle=True,
    #     num_workers=0,
    #     batch_size=2
    # )
    # print(len(val_dataset))
    # data, target = next(iter(val_loader))
    # print(data.shape, target.shape)
    # print(data.dtype)

    
   
