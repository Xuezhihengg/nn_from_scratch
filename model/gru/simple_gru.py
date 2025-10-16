import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 50, hidden_size = 100):
        super(SimpleGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 初始化权重
        self.W_hr = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xr = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hz = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xz = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xh = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(self.hidden_size))

        # 输出层线性变换
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, h_prev = None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
            h_prev: tensor, shape(batch_size, hidden_size)
        """
        batch_size, T = inputs.size()
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device = inputs.device)
        
        outputs = []
        h_t = h_prev
        for t in range(T):
            # 取第t个时间步所有batch的数据： (batch_size,)
            x_t = inputs[:, t]
            x_t = self.embedding(x_t)   # (batch_size, embedding_dim)

            R_t = torch.sigmoid(h_t @ self.W_hr + x_t @ self.W_xr + self.b_r)
            Z_t = torch.sigmoid(h_t @ self.W_hz + x_t @ self.W_xz + self.b_z)

            h_tilda = torch.tanh((R_t * h_t) @ self.W_hh + x_t @ self.W_xh + self.b_h)
            h_t = Z_t * h_tilda + (1 - Z_t) * h_t

            y_t = self.fc(h_t)
            outputs.append(y_t)     # len = t, 每个元素为tensor, shape(batch_size, vocab_size)

        outputs = torch.stack(outputs, dim = 1)     # (batch_size, T, vocab_size)

        return outputs











