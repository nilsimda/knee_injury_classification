import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MRNetDataset(Dataset):
    def __init__(self, data_path, labels_path):

        self.data_path = data_path
        self.data = sorted(os.listdir(data_path))

        def get_multiclass_labels(phase='train'):
            abnormal_df = pd.read_csv(os.path.join(labels_path, phase+'-abnormal.csv'), header=None)
            acl_df = pd.read_csv(os.path.join(labels_path, phase+'-acl.csv'), header=None)
            meniscus_df = pd.read_csv(os.path.join(labels_path, phase+'-meniscus.csv'), header=None)

            abnormal_df.loc[acl_df[1] == 1, 1] = 2
            abnormal_df.loc[meniscus_df[1] == 1, 1] = 3
            return abnormal_df[1].to_list()

        if data_path.endsWith('train'):
            self.labels = get_multiclass_labels()            
        elif data_path.endsWith('valid'):
            self.labels = get_multiclass_labels('valid') 
        else:
            raise ValueError('data_path must end with either train or valid')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_path = os.path.join(self.data_path, self.data[idx])
        return np.load(item_path), self.labels[idx]

