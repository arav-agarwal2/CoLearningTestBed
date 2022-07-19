import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


def get_dataloader(path, batch_size=32, num_workers=4):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        f.close()
    except Exception as ex:
        print("Error during unpickling object", ex)
        exit()
    num_samples = len(data['label'])
    train_split = num_samples // 10 * 8
    valid_split = train_split + num_samples // 10
    train_data = {k:v[:train_split] for (k,v) in data.items()}
    print("Train data: {}".format(train_data['label'].shape[0]))
    valid_data = {k:v[train_split:valid_split] for (k,v) in data.items()}
    print("Valid data: {}".format(valid_data['label'].shape[0]))
    test_data = {k:v[valid_split:] for (k,v) in data.items()}
    print("Test data: {}".format(test_data['label'].shape[0]))

    traindata = DataLoader(SyntheticDataset(train_data,),
                    shuffle=True, 
                    num_workers=num_workers, 
                    batch_size=batch_size, 
                    collate_fn=process_input)
    validdata = DataLoader(SyntheticDataset(valid_data,),
                        shuffle=False, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        collate_fn=process_input)
    testdata = DataLoader(SyntheticDataset(test_data),
                        shuffle=False, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        collate_fn=process_input)

    return traindata, validdata, testdata


class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        return (torch.tensor(self.data['a'][index]), torch.tensor(self.data['b'][index]), torch.tensor(self.data['label'][index]))


def process_input(inputs):
    processed_input = []
    labels = []

    for i in range(len(inputs[0])-1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input.append(torch.stack(feature))

    for sample in inputs:  
        labels.append(sample[-1])
        
    return processed_input[0], processed_input[1], torch.tensor(labels).view(len(inputs), 1)
