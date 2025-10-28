import torch
import torch.nn as nn

class BidGRU(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super(BidGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 正向GRU参数
        self.W_hr_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xr_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_r_forward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hz_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xz_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_z_forward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hh_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xh_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_h_forward = nn.Parameter(torch.zeros(hidden_size))

        # 反向GRU参数
        self.W_hr_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xr_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_r_backward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hz_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xz_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_z_backward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hh_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xh_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_h_backward = nn.Parameter(torch.zeros(hidden_size))


    def forward(self, inputs, h_prev_forward=None, h_prev_backward=None):
        """
        Args:
            inputs: tensor, shape (batch_size, seq_length, input_size)
            h_prev_forward: tensor, (batch_size, hidden_size) or None
            h_prev_backward: tensor, (batch_size, hidden_size) or None
        Returns:
            h_stitched: tensor, (batch_size, seq_length, hidden_size * 2)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev_forward is None:
            h_t_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t_forward = h_prev_forward

        if h_prev_backward is None:
            h_t_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t_backward = h_prev_backward

        # 正向GRU计算
        h_forward = []
        for t in range(T):
            x_t = inputs[:, t, :]  # (batch_size, input_size)

            R_t = torch.sigmoid(h_t_forward @ self.W_hr_forward + x_t @ self.W_xr_forward + self.b_r_forward)
            Z_t = torch.sigmoid(h_t_forward @ self.W_hz_forward + x_t @ self.W_xz_forward + self.b_z_forward)
            h_tilda = torch.tanh((R_t * h_t_forward) @ self.W_hh_forward + x_t @ self.W_xh_forward + self.b_h_forward)
            h_t_forward = Z_t * h_tilda + (1 - Z_t) * h_t_forward

            h_forward.append(h_t_forward)

        # 反向GRU计算
        h_backward = []
        for t in range(T - 1, -1, -1):
            x_t = inputs[:, t, :]  # (batch_size, input_size)

            R_t = torch.sigmoid(h_t_backward @ self.W_hr_backward + x_t @ self.W_xr_backward + self.b_r_backward)
            Z_t = torch.sigmoid(h_t_backward @ self.W_hz_backward + x_t @ self.W_xz_backward + self.b_z_backward)
            h_tilda = torch.tanh((R_t * h_t_backward) @ self.W_hh_backward + x_t @ self.W_xh_backward + self.b_h_backward)
            h_t_backward = Z_t * h_tilda + (1 - Z_t) * h_t_backward

            h_backward.insert(0, h_t_backward)  # 保持时间顺序一致

        h_forward = torch.stack(h_forward, dim=1)   # (batch_size, T, hidden_size)
        h_backward = torch.stack(h_backward, dim=1) # (batch_size, T, hidden_size)

        h_stitched = torch.cat([h_forward, h_backward], dim=2) # (batch_size, T, hidden_size * 2)

        return h_stitched