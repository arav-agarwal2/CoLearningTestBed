import torch
import pickle
from torch.utils.data import Dataset, DataLoader


def get_dataloader(path, keys=['a','b','label'], modalities=[0,1], batch_size=32, num_workers=4):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object", ex)
        exit()
    label = keys[-1]
    print("Train data: {}".format(data['train'][label].shape[0]))
    print("Test data: {}".format(data['test'][label].shape[0]))

    traindata = DataLoader(SyntheticDataset(data['train'], keys, modalities=modalities),
                    shuffle=True, 
                    num_workers=num_workers, 
                    batch_size=batch_size, 
                    collate_fn=process_input)
    # validdata = DataLoader(SyntheticDataset(valid_data, keys, modalities=modalities),
    #                     shuffle=False, 
    #                     num_workers=num_workers, 
    #                     batch_size=batch_size, 
    #                     collate_fn=process_input)
    validdata = []
    testdata = DataLoader(SyntheticDataset(data['test'], keys, modalities=modalities),
                        shuffle=False, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        collate_fn=process_input)

    return traindata, validdata, testdata


class SyntheticDataset(Dataset):
    def __init__(self, data, keys, modalities):
        self.data = data
        self.keys = keys
        self.modalities = modalities
        
    def __len__(self):
        return len(self.data[self.keys[-1]])

    def __getitem__(self, index):
        tmp = []
        for i, modality in enumerate(self.modalities):
            if modality:
                tmp.append(torch.tensor(self.data[self.keys[i]][index]))
            else:
                tmp.append(torch.ones(self.data[self.keys[i]][index].size))
        tmp.append(torch.tensor(self.data[self.keys[-1]][index]))
        return tmp


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
        
    return processed_input, torch.tensor(labels).view(len(inputs), 1)
