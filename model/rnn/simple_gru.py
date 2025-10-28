import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super(SimpleGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 门控和更新门权重
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xr = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))

        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xz = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_size))

        # 候选隐藏状态权重
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs, h_prev=None):
        """
        Args:
            inputs: tensor, shape (batch_size, seq_length, input_size)
            h_prev: tensor, shape (batch_size, hidden_size)
        Returns:
            outputs: tensor, shape (batch_size, seq_length, hidden_size)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t = h_prev

        outputs = []
        for t in range(T):
            x_t = inputs[:, t, :]  # (batch_size, input_size)

            R_t = torch.sigmoid(h_t @ self.W_hr + x_t @ self.W_xr + self.b_r)
            Z_t = torch.sigmoid(h_t @ self.W_hz + x_t @ self.W_xz + self.b_z)

            h_tilda = torch.tanh((R_t * h_t) @ self.W_hh + x_t @ self.W_xh + self.b_h)
            h_t = Z_t * h_tilda + (1 - Z_t) * h_t

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_length, hidden_size)

        return outputs
    
    def forward_step(self, x_t, h_prev):
        R_t = torch.sigmoid(h_prev @ self.W_hr + x_t @ self.W_xr + self.b_r)
        Z_t = torch.sigmoid(h_prev @ self.W_hz + x_t @ self.W_xz + self.b_z)
        h_tilda = torch.tanh((R_t * h_prev) @ self.W_hh + x_t @ self.W_xh + self.b_h)
        h_t = Z_t * h_tilda + (1 - Z_t) * h_prev
        return h_t