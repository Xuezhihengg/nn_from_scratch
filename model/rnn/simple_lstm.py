import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化权重参数
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xi = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xf = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xo = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.W_hc = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xc = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs, h_prev=None):
        """
        inputs: Tensor, shape (batch_size, seq_length, input_size)
        h_prev: tuple of (h_t, c_t), or None
            h_t, c_t: (batch_size, hidden_size)
        返回：
            outputs: (batch_size, seq_length, hidden_size)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t, c_t = h_prev

        outputs = []

        for t in range(T):
            x_t = inputs[:, t, :]

            I_t = torch.sigmoid(h_t @ self.W_hi + x_t @ self.W_xi + self.b_i)
            F_t = torch.sigmoid(h_t @ self.W_hf + x_t @ self.W_xf + self.b_f)
            O_t = torch.sigmoid(h_t @ self.W_ho + x_t @ self.W_xo + self.b_o)
            C_tilde = torch.tanh(h_t @ self.W_hc + x_t @ self.W_xc + self.b_c)

            c_t = F_t * c_t + I_t * C_tilde
            h_t = O_t * torch.tanh(c_t)

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return outputs