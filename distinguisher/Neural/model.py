# the scrip is collected from https://github.com/RaffaeleGalliera/pytorch-cnn-text-classification/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc(x)  # (N, C)
        return logit

    @staticmethod
    def __count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RNN(nn.Module):
    def __init__(self,  vocab_size, embedding_dim, hidden_size, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_size)

        self.linear = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        seq_len , batch_size = x.shape
        vec = self.embedding(x)
        output,hidden = self.rnn(vec)
        out = self.linear(hidden.view(batch_size, -1))
        return out

    @staticmethod
    def __count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LSTM(nn.Module):
    def __init__(self,  vocab_size, embedding_dim, hidden_size, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_dim)
    
    def forward(self, x):
        vec = self.embedding(x)
        _, (hidden, cell) = self.lstm(vec)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.fc(hidden)
        return output

    @staticmethod
    def __count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)