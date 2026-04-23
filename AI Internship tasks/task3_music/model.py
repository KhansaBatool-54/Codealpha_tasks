# model.py — LSTM Model banao PyTorch se

import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MusicLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        # LSTM layer — music patterns seekhne ke liye
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Dropout — overfitting rokne ke liye
        self.dropout = nn.Dropout(0.3)
        
        # Output layer — next note predict karne ke liye
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out