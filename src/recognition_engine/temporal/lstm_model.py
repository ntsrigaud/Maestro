import torch
import torch.nn as nn

class BiLSTMAttention(nn.Module):
    """
    Bi-directional LSTM with attention for temporal modeling of ViT features.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        out = self.classifier(context)  # (batch, num_classes)
        return out, attn_weights.squeeze(-1)
