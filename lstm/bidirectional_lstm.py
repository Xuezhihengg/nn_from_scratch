import torch
import torch.nn as nn

class BidLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100):
        super(BidLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 正向LSTM参数
        self.W_hi_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xi_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_i_forward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hf_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xf_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_f_forward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_ho_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xo_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_o_forward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hc_forward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xc_forward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_c_forward = nn.Parameter(torch.zeros(self.hidden_size))

        # 反向LSTM参数
        self.W_hi_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xi_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_i_backward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hf_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xf_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_f_backward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_ho_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xo_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_o_backward = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hc_backward = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xc_backward = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_c_backward = nn.Parameter(torch.zeros(self.hidden_size))

        # 输出层，接收正向和反向hidden拼接后隐藏状态
        self.fc = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, inputs, h_prev_forward=None, h_prev_backward=None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
            h_prev_forward: tuple of tensors (h_0, c_0), each (batch_size, hidden_size)
            h_prev_backward: tuple of tensors (h_0, c_0), each (batch_size, hidden_size)
        Return:
            outputs: (batch_size, T, vocab_size)
        """
        batch_size, T = inputs.size()

        # 初始化正向隐藏状态和细胞状态
        if h_prev_forward is None:
            h_t_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            c_t_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t_forward, c_t_forward = h_prev_forward

        # 初始化反向隐藏状态和细胞状态
        if h_prev_backward is None:
            h_t_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            c_t_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t_backward, c_t_backward = h_prev_backward

        h_forward = []
        for t in range(T):
            x_t = inputs[:, t]
            x_t = self.embedding(x_t)

            i_t = torch.sigmoid(h_t_forward @ self.W_hi_forward + x_t @ self.W_xi_forward + self.b_i_forward)
            f_t = torch.sigmoid(h_t_forward @ self.W_hf_forward + x_t @ self.W_xf_forward + self.b_f_forward)
            o_t = torch.sigmoid(h_t_forward @ self.W_ho_forward + x_t @ self.W_xo_forward + self.b_o_forward)
            c_hat_t = torch.tanh(h_t_forward @ self.W_hc_forward + x_t @ self.W_xc_forward + self.b_c_forward)

            c_t_forward = f_t * c_t_forward + i_t * c_hat_t
            h_t_forward = o_t * torch.tanh(c_t_forward)

            h_forward.append(h_t_forward)

        h_backward = []
        for t in range(T - 1, -1, -1):
            x_t = inputs[:, t]
            x_t = self.embedding(x_t)

            i_t = torch.sigmoid(h_t_backward @ self.W_hi_backward + x_t @ self.W_xi_backward + self.b_i_backward)
            f_t = torch.sigmoid(h_t_backward @ self.W_hf_backward + x_t @ self.W_xf_backward + self.b_f_backward)
            o_t = torch.sigmoid(h_t_backward @ self.W_ho_backward + x_t @ self.W_xo_backward + self.b_o_backward)
            c_hat_t = torch.tanh(h_t_backward @ self.W_hc_backward + x_t @ self.W_xc_backward + self.b_c_backward)

            c_t_backward = f_t * c_t_backward + i_t * c_hat_t
            h_t_backward = o_t * torch.tanh(c_t_backward)

            h_backward.insert(0, h_t_backward)  # 反向时间步，保持顺序一致

        # 将正向和反向隐藏状态堆叠、拼接
        h_forward = torch.stack(h_forward, dim=1)    # (batch_size, T, hidden_size)
        h_backward = torch.stack(h_backward, dim=1)  # (batch_size, T, hidden_size)

        h_stitched = torch.cat([h_forward, h_backward], dim=2)  # (batch_size, T, hidden_size*2)

        outputs = self.fc(h_stitched)

        return outputs