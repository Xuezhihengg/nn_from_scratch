import torch.nn as nn
from .base_rnn import RNN, GRU, LSTM

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = LSTM(embedding_dim, hidden_size, num_layers, bidirectional)
        linear_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(linear_input_size, vocab_size)
    
    def forward(self, inputs, hidden=None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
        Return:
            logits: tensor, shape(batch_size, seq_length, vocab_size)
        """
        embed = self.embedding(inputs)  # (batch_size, seq_length, embedding_dim)
        output = self.rnn(embed, hidden)  # (batch_size, seq_length, hidden_size)
        logits = self.fc(output[0])   # (batch_size, seq_length, vocab_size)
        return logits
