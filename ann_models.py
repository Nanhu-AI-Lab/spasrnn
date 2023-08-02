'''
make a comparison with LSTM and GRU
'''
import torch
from torch import nn

device = 'cuda'

class LSTM(nn.Module):
    '''
    switch to LSTM
    '''
    def __init__(self, glove_matrix, em_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        #self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.bi = False
        # [0-10001] => [100]
        self.embedding = nn.Embedding(glove_matrix.shape[0], em_dim)
        #self.embedding.load_state_dict({'weight': glove_matrix})
        # [100] => [256]
        self.lstm = nn.LSTM(em_dim, hidden_dim, num_layers, \
                            batch_first=True, bidirectional=self.bi)
        print(f'Bi-LSTM mode is { self.bi}')
        if self.bi:
            self.fc = nn.Linear(hidden_dim*2, out_dim)
        else:
            self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [seq, b, 1]
        # [seq, b, 1] => [seq, b, 100]
        x = self.dropout(self.embedding(x))
        # Set initial hidden and cell states
        if self.bi:
            h0 = torch.zeros(self.num_layers*2, x.size(0), \
                            self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), \
                            self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), \
                            self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), \
                            self.hidden_size).to(device)
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    '''
    switch to GRU
    '''
    def __init__(self, glove_matrix, em_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.bi = False
        # [0-10001] => [100]
        self.embedding = nn.Embedding(glove_matrix.shape[0], em_dim)
        #self.embedding.load_state_dict({'weight': glove_matrix})
        # [100] => [256]
        self.gru = nn.GRU(em_dim, hidden_dim, num_layers, \
                          batch_first=True, bidirectional=self.bi)
        print(f'Bi-GRU mode is {self.bi}')
        if self.bi:
            self.fc = nn.Linear(hidden_dim*2, out_dim)
        else:
            self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [seq, b, 1]
        # [seq, b, 1] => [seq, b, 100]
        x = self.dropout(self.embedding(x))
        # Set initial hidden and cell states
        if self.bi:
            h0 = torch.zeros(self.num_layers*2, x.size(0), \
                            self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), \
                            self.hidden_size).to(device)
        # Forward propagate GRU
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.gru(x, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
