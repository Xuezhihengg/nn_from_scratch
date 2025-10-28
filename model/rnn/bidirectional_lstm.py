import torch
import torch.nn as nn


class BidLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super(BidLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 正向LSTM参数
        self.W_hi_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xi_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_i_forward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hf_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xf_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_f_forward = nn.Parameter(torch.zeros(hidden_size))

        self.W_ho_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xo_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_o_forward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hc_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xc_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_c_forward = nn.Parameter(torch.zeros(hidden_size))

        # 反向LSTM参数
        self.W_hi_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xi_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_i_backward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hf_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xf_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_f_backward = nn.Parameter(torch.zeros(hidden_size))

        self.W_ho_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xo_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_o_backward = nn.Parameter(torch.zeros(hidden_size))

        self.W_hc_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_xc_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.b_c_backward = nn.Parameter(torch.zeros(hidden_size))


    def forward(self, inputs, h_prev_forward=None, c_prev_forward=None, h_prev_backward=None, c_prev_backward=None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length, input_size)
            h_prev_forward: tensor(batch_size, hidden_size) or None
            c_prev_forward: tensor(batch_size, hidden_size) or None
            h_prev_backward: tensor(batch_size, hidden_size) or None
            c_prev_backward: tensor(batch_size, hidden_size) or None

        Return:
            h_stitched: tensor, shape(batch_size, seq_length, hidden_size*2)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        # 初始化正向hidden和cell状态
        if h_prev_forward is None:
            h_t_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t_forward = h_prev_forward

        if c_prev_forward is None:
            c_t_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            c_t_forward = c_prev_forward

        # 初始化反向hidden和cell状态
        if h_prev_backward is None:
            h_t_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t_backward = h_prev_backward

        if c_prev_backward is None:
            c_t_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            c_t_backward = c_prev_backward

        h_forward = []
        for t in range(T):
            x_t = inputs[:, t, :]  # (batch_size, input_size)

            i_t = torch.sigmoid(h_t_forward @ self.W_hi_forward + x_t @ self.W_xi_forward + self.b_i_forward)
            f_t = torch.sigmoid(h_t_forward @ self.W_hf_forward + x_t @ self.W_xf_forward + self.b_f_forward)
            o_t = torch.sigmoid(h_t_forward @ self.W_ho_forward + x_t @ self.W_xo_forward + self.b_o_forward)
            c_hat_t = torch.tanh(h_t_forward @ self.W_hc_forward + x_t @ self.W_xc_forward + self.b_c_forward)

            c_t_forward = f_t * c_t_forward + i_t * c_hat_t
            h_t_forward = o_t * torch.tanh(c_t_forward)

            h_forward.append(h_t_forward)

        h_backward = []
        for t in range(T - 1, -1, -1):
            x_t = inputs[:, t, :]  # (batch_size, input_size)

            i_t = torch.sigmoid(h_t_backward @ self.W_hi_backward + x_t @ self.W_xi_backward + self.b_i_backward)
            f_t = torch.sigmoid(h_t_backward @ self.W_hf_backward + x_t @ self.W_xf_backward + self.b_f_backward)
            o_t = torch.sigmoid(h_t_backward @ self.W_ho_backward + x_t @ self.W_xo_backward + self.b_o_backward)
            c_hat_t = torch.tanh(h_t_backward @ self.W_hc_backward + x_t @ self.W_xc_backward + self.b_c_backward)

            c_t_backward = f_t * c_t_backward + i_t * c_hat_t
            h_t_backward = o_t * torch.tanh(c_t_backward)

            h_backward.insert(0, h_t_backward)  # 保持时间步顺序一致

        h_forward = torch.stack(h_forward, dim=1)  # (batch_size, T, hidden_size)
        h_backward = torch.stack(h_backward, dim=1)  # (batch_size, T, hidden_size)

        h_stitched = torch.cat([h_forward, h_backward], dim=2)  # (batch_size, T, hidden_size*2)

        return h_stitched