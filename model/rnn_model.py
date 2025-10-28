import torch.nn as nn
from .rnn import SimpleRNN

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = SimpleRNN(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, inputs, hidden=None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
        Return:
            logits: tensor, shape(batch_size, seq_length, vocab_size)
        """
        embed = self.embedding(inputs)  # (batch_size, seq_length, embedding_dim)
        output = self.rnn(embed, hidden)  # (batch_size, seq_length, hidden_size)
        logits = self.fc(output)   # (batch_size, seq_length, vocab_size)
        return logits
