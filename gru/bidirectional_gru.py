import torch
import torch.nn as nn


class BidGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100):
        super(BidGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 正向GRU参数
        self.W_hr_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xr_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_r_forward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hz_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xz_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_z_forward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hh_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xh_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_h_forward = nn.Parameter(torch.zeros(self.hidden_size))

        # 反向GRU参数
        self.W_hr_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xr_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_r_backward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hz_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xz_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_z_backward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hh_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xh_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_h_backward = nn.Parameter(torch.zeros(self.hidden_size))

        # 输出层，接收正向和反向hidden拼接后隐藏状态
        self.fc = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, inputs, h_prev_forward=None, h_prev_backward=None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
            h_prev_forward: tensor, shape(batch_size, hidden_size)
            h_prev_backward: tensor, shape(batch_size, hidden_size)
        Return:
            outputs: (batch_size, T, vocab_size)
        """
        batch_size, T = inputs.size()

        if h_prev_forward is None:
            h_prev_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if h_prev_backward is None:
            h_prev_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        # 正向GRU计算
        h_forward = []
        h_t_forward = h_prev_forward

        for t in range(T):
            x_t = self.embedding(inputs[:, t])  # (batch_size, embedding_dim)

            R_t = torch.sigmoid(h_t_forward @ self.W_hr_forward + x_t @ self.W_xr_forward + self.b_r_forward)
            Z_t = torch.sigmoid(h_t_forward @ self.W_hz_forward + x_t @ self.W_xz_forward + self.b_z_forward)
            h_tilda = torch.tanh((R_t * h_t_forward) @ self.W_hh_forward + x_t @ self.W_xh_forward + self.b_h_forward)
            h_t_forward = Z_t * h_tilda + (1 - Z_t) * h_t_forward

            h_forward.append(h_t_forward)

        # 反向GRU计算
        h_backward = []
        h_t_backward = h_prev_backward

        for t in range(T - 1, -1, -1):
            x_t = self.embedding(inputs[:, t])

            R_t = torch.sigmoid(h_t_backward @ self.W_hr_backward + x_t @ self.W_xr_backward + self.b_r_backward)
            Z_t = torch.sigmoid(h_t_backward @ self.W_hz_backward + x_t @ self.W_xz_backward + self.b_z_backward)
            h_tilda = torch.tanh((R_t * h_t_backward) @ self.W_hh_backward + x_t @ self.W_xh_backward + self.b_h_backward)
            h_t_backward = Z_t * h_tilda + (1 - Z_t) * h_t_backward

            h_backward.insert(0, h_t_backward)  # 反向序列，先放到0位置保证顺序与正向对应

        # 拼接正向和反向隐藏状态
        h_forward = torch.stack(h_forward, dim=1)   # (batch_size, T, hidden_size)
        h_backward = torch.stack(h_backward, dim=1) # (batch_size, T, hidden_size)

        h_stitched = torch.cat([h_forward, h_backward], dim=2)  # (batch_size, T, hidden_size * 2)

        outputs = self.fc(h_stitched)  # (batch_size, T, vocab_size)

        return outputs