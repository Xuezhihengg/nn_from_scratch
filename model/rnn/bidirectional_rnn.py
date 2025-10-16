import torch
import torch.nn as nn

class BidRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 50, hidden_size = 100):
        super(BidRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.W_xh_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.W_hh_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.b_h_forward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_xh_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.W_hh_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.b_h_backward = nn.Parameter(torch.zeros(self.hidden_size))

        self.fc = nn.Linear(self.hidden_size * 2, self.vocab_size)

    
    def forward(self, inputs, h_prev = None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
            h_prev: tensor, shape(batch_size, hidden_size)
        Return:
            outputs: (batch_size, T, vocab_size)
        """
        batch_size, T = inputs.size()

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device = inputs.device)
        
        h_forward = []
        h_backward = []

        # 正向计算
        h_t = h_prev
        for t in range(T):
            x_t = self.embedding(inputs[:, t])      # (batch_size, embedding_dim)
            h_t = torch.tanh(h_t @ self.W_hh_forward + x_t @ self.W_xh_forward + self.b_h_forward)
            h_forward.append(h_t)       # len = T, 每个元素shape(batch_size, hidden_size)

        # 反向计算
        h_t = h_prev
        for t in range(T-1, -1, -1):
            x_t = self.embedding(inputs[:, t])      # (batch_size, embedding_dim)
            h_t = torch.tanh(h_t @ self.W_hh_backward + x_t @ self.W_xh_backward + self.b_h_backward)
            h_backward.insert(0, h_t)   # len = T, 每个元素shape(batch_size, hidden_size)

        # 拼接正反向隐藏状态
        h_forward = torch.stack(h_forward, dim = 1) # (batch_size, T, hidden_size)
        h_backward = torch.stack(h_backward, dim = 1) # (batch_size, T, hidden_size)
        h_stitched = torch.cat([h_forward, h_backward], dim = 2)  # (batch_size, T, hidden_size * 2)

        # 计算y
        outputs = self.fc(h_stitched)   # (batch_size, T, vocab_size)

        return outputs




        