from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import torchtext as text
import torch
import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
from numpy.core.numeric import zeros_like
from torch.nn.functional import pad
from torch.nn import functional as F

sys.path.append(os.getcwd())


np.seterr(divide='ignore', invalid='ignore')


def z_norm(dataset, max_seq_len=50):
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


def get_rawtext(path, data_kind, vids):
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


def get_word2id(text_data, vids):
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['unk']
    data_processed = dict()
    for i, segment in enumerate(text_data):
        words = []
        _words = segment.split()
        for word in _words:
            words.append(word2id[word])
        words = np.asarray(words)
        data_processed[vids[i]] = words

    def return_unk():
        return UNK

    word2id.default_factory = return_unk
    return data_processed, word2id


def get_word_embeddings(word2id, save=False):
    vec = text.vocab.GloVe(name='840B', dim=300)
    tokens = []
    for w, _ in word2id.items():
        tokens.append(w)

    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
    return ret


def glove_embeddings(text_data, vids, paddings=50):
    data_prod, w2id = get_word2id(text_data, vids)
    word_embeddings_looks_up = get_word_embeddings(w2id)
    looks_up = word_embeddings_looks_up.numpy()
    embedd_data = []
    for vid in vids:
        d = data_prod[vid]
        tmp = []
        look_up = [looks_up[x] for x in d]
        # Padding with zeros at the front
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings - len(d)):
                tmp.append(np.zeros(300, ))
            for x in d:
                tmp.append(looks_up[x])

        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


class MOSI(Dataset):

    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None, max_pad=False, max_pad_num=50, z_norm=False) -> None:
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
                # start = 0
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            vision = torch.nan_to_num(
                (vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
            audio = torch.nan_to_num(
                (audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
            text = torch.nan_to_num(
                (text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        def get_class(flag):
            if flag > 0:
                return [[1]]
            else:
                return [[0]]

        tmp_label = self.dataset['labels'][ind]
        tmp_label = self.dataset['labels'][ind]

        label = torch.tensor(get_class(tmp_label)).long()

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind,
                    label]
        else:
            if self.max_pad:
                tmp = [vision, audio, text, label]
                for i in range(len(tmp) - 1):
                    tmp[i] = tmp[i][:self.max_pad_num]
                    tmp[i] = F.pad(
                        tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))
            else:
                tmp = [vision, audio, text, ind, label]
            return tmp

    def __len__(self):
        return self.dataset['vision'].shape[0]


def get_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, train_shuffle: bool = True,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, z_norm=False) -> DataLoader:
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}

    process = eval("process_2") if max_pad else eval("process_1")

    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]
    # processed_dataset['test']['vision'] = np.zeros(alldata['test']['vision'].shape)
    # processed_dataset['test']['audio'] = np.zeros(alldata['test']['audio'].shape)

    train = DataLoader(MOSI(processed_dataset['train'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, z_norm=z_norm),
                       shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size,
                       collate_fn=process)
    valid = DataLoader(MOSI(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, z_norm=z_norm),
                       shuffle=False, num_workers=num_workers, batch_size=batch_size,
                       collate_fn=process)

    # Only keep text modality
    test = DataLoader(MOSI(processed_dataset['test'], flatten_time_series, task=task, max_pad=max_pad,
                           max_pad_num=max_seq_len, z_norm=z_norm), shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process)

    print("Dataset split")
    print("Train Set: {}".format(len(train)))
    print("Validation Set: {}".format(len(valid)))
    print("Test Set: {}".format(len(test)))
            
    return train, valid, test


def process_1(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(
            torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:

        inds.append(sample[-2])
        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        if sample[-1].shape[1] > 1:
            labels.append(
                sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
        torch.tensor(inds).view(len(inputs), 1), torch.tensor(
            labels).view(len(inputs), 1)


def process_2(inputs: List):
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(
            torch.as_tensor([v.size(0) for v in feature]))
        # pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        if sample[-1].shape[1] > 1:
            labels.append(
                sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input[0], processed_input[1], processed_input[2], torch.tensor(labels).view(len(inputs), 1)