import numpy as np
import itertools, json, os, logging
import dill
import torch, random

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torchtext.legacy import data

class MemSeqDataset(data.Dataset):
    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs) -> None:
        
        text_field.tokenizer = lambda x: x.split()
        fields = [('text', text_field), ('label', label_field)]
        raw_labels = []
        raw_examples = []
        if examples is None:
            examples = []
            with open(path) as f:
                line = f.readline()
                while line:
                    if "__label__1" == line[:10]:
                        label = "0"
                    elif "__label__2" == line[:10]:
                        label = "1"
                    else:
                        raise RuntimeError()
                    text = line[11:]
                    raw_labels.append(label)
                    raw_examples.append(text)
                    line = f.readline()
            for i in range(len(raw_examples)):
                examples.append(data.Example.fromlist([raw_examples[i], raw_labels[i]], fields))
        random.shuffle(examples)
            
        super(MemSeqDataset, self).__init__(examples, fields, **kwargs)
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    
    @classmethod
    def splits(cls, text_field, label_field, path, dev_ratio=.05, shuffle=True, root='.', **kwargs):
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


def get_train_dev_test_loader(text_field, label_field, train_path, test_path, batch_size, **kargs):
    train, dev = MemSeqDataset.splits(text_field, label_field, path=train_path)
    test = MemSeqDataset(text_field, label_field, path=test_path)

    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                (train, dev, test), 
                                batch_sizes=(batch_size, batch_size*2, batch_size*2),
                                **kargs)
    return train_iter, dev_iter, test_iter

def get_test_loader(text_field, label_field, test_path, batch_size, **kargs):
    test = MemSeqDataset(text_field, label_field, path=test_path)
    return data.Iterator(test, batch_size, **kargs)

def save_model_and_fields(model, model_path, text_field, label_field, data_folder):
    torch.save(model.state_dict(), model_path)
    text_field_path = os.path.join(data_folder, "text.bin")
    label_field_path = os.path.join(data_folder, "label.bin")
    with open(text_field_path, "wb") as out:
        dill.dump(text_field, out)
    with open(label_field_path, "wb") as out:
        dill.dump(label_field, out)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def load_fields(data_folder):
    text_field_path = os.path.join(data_folder, "text.bin")
    label_field_path = os.path.join(data_folder, "label.bin")
    with open(text_field_path, "rb") as in_file:
        text_field = dill.load(in_file)
    with open(label_field_path, "rb") as in_file:
        label_field = dill.load(in_file)
    return text_field, label_field

def output_result(accuracy, size, file_path):
    with open(file_path, "w") as f:
        f.write(f"{size}, {accuracy}")

class ModelConfig:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
        self.config_file = os.path.join(data_folder, "config.json")
        with open(self.config_file) as f:
            config = json.load(f)
        self.train_file = os.path.join(data_folder, config["train_file"])
        self.val_file = os.path.join(data_folder, config["val_file"])
        self.is_train = config["phase"] == "train" # train or val
        self.batch_size = config["batch_size"]
        self.model_path = os.path.join(data_folder, config["model_path"]) # path to save/load model
        self.model_arch = config["model_arch"] # LSTM, GRU, CNN
        self.model_args = config["model_args"]
        self.result_path = os.path.join(data_folder, config["output_path"])
