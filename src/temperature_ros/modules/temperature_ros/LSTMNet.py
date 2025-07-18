# Long-Short Term Memory Neural Network class
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LSTMNet(nn.Module):
    def __init__(self, num_features=1, num_hidden1=125, num_hidden2=100, num_classes=6, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=num_hidden1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=num_hidden1, hidden_size=num_hidden2, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(num_hidden2, num_classes)

    
    def forward(self, x, lengths = None):
        # x: (batch, seq_len, num_features)
        # lengths: (batch,)  -- original lengths before padding
        if lengths is None:
            # x shape: (batch, seq_len, num_features)
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            # Only take last time step output from second LSTM
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            out = out[:, -1, :]  # last time step
            out = self.fc(out)
        else:
            # Pack the padded sequence
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # LSTM 1
            out_packed, _ = self.lstm1(x_packed)
            out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
            out_padded = self.dropout1(out_padded)

            # Pack again before 2nd LSTM
            out_packed = pack_padded_sequence(out_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, (h_n, c_n) = self.lstm2(out_packed)
            out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
            out_padded = self.dropout2(out_padded)

            # Use last valid time step for each sequence
            # (lengths - 1) because of 0-indexing
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out_padded.size(2))
            last_outputs = out_padded.gather(1, idx).squeeze(1)  # shape: (batch, hidden_size)

            out = self.fc(last_outputs)

        return out                         # use linear output if using CrossEntropyLoss
        #return F.log_softmax(out, dim=1)  # use log_softmax if using NLLLoss

        
"""
    def __init__(self, num_features=1, num_hidden1=125, num_hidden2=100, num_classes=6, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_hidden1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(num_hidden1, num_classes)
    def forward(self, x, lengths = None):
        # x: (batch, seq_len, num_features)
        # lengths: (batch,)  -- original lengths before padding
        if lengths is None:
            # x shape: (batch, seq_len, num_features)
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # last time step
            out = self.fc(out)
        else:
            # Pack the padded sequence
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # LSTM 1
            out_packed, _ = self.lstm(x_packed)
            out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

            # (lengths - 1) because of 0-indexing
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out_padded.size(2))
            last_outputs = out_padded.gather(1, idx).squeeze(1)  # shape: (batch, hidden_size)

            out = self.fc(last_outputs)

        return out                         # use linear output if using CrossEntropyLoss
        #return F.log_softmax(out, dim=1)  # use log_softmax if using NLLLoss


class LSTMNet(nn.Module):

    def __init__(self, num_features=1, num_hidden1=125, num_hidden2=100, num_classes=6, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.timeseries_lenght = num_features
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=num_hidden1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=num_hidden1, hidden_size=num_hidden2, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(num_hidden2, num_classes)

#old forward
#    def forward(self, x):
#        # x shape: (batch, seq_len, num_features)
#        out, _ = self.lstm1(x)
#        out = self.dropout1(out)
#        # Only take last time step output from second LSTM
#        out, _ = self.lstm2(out)
#        out = self.dropout2(out)
#        out = out[:, -1, :]  # last time step
#        out = self.fc(out)
#        return out                         # use linear output if using CrossEntropyLoss
#        #return F.log_softmax(out, dim=1)  # use log_softmax if using NLLLoss
    def forward(self, x, lengths = None):
        # x: (batch, seq_len, num_features)
        # lengths: (batch,)  -- original lengths before padding
        if lengths is None:
            # x shape: (batch, seq_len, num_features)
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            # Only take last time step output from second LSTM
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            out = out[:, -1, :]  # last time step
            out = self.fc(out)
        else:
            # Pack the padded sequence
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

            # LSTM 1
            out_packed, _ = self.lstm1(x_packed)
            out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
            out_padded = self.dropout1(out_padded)

            # Pack again before 2nd LSTM
            out_packed = pack_padded_sequence(out_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, (h_n, c_n) = self.lstm2(out_packed)
            out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
            out_padded = self.dropout2(out_padded)

            # Use last valid time step for each sequence
            # (lengths - 1) because of 0-indexing
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out_padded.size(2))
            last_outputs = out_padded.gather(1, idx).squeeze(1)  # shape: (batch, hidden_size)

            out = self.fc(last_outputs)

        return out                         # use linear output if using CrossEntropyLoss
        #return F.log_softmax(out, dim=1)  # use log_softmax if using NLLLoss    
"""