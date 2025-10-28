import torch
import torch.nn as nn

class BidRNN(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super(BidRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 正向参数
        self.W_xh_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h_forward = nn.Parameter(torch.zeros(hidden_size))

        # 反向参数
        self.W_xh_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h_backward = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs, h_prev_forward=None, h_prev_backward=None):
        """
        Args:
            inputs: Tensor, shape (batch_size, seq_len, input_size)
            h_prev_forward: None or Tensor (batch_size, hidden_size)
            h_prev_backward: None or Tensor (batch_size, hidden_size)
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, hidden_size*2)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev_forward is None:
            h_prev_forward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        if h_prev_backward is None:
            h_prev_backward = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        # 正向隐藏状态序列
        h_forward = []
        h_t = h_prev_forward
        for t in range(T):
            x_t = inputs[:, t, :]  # (batch_size, input_size)
            h_t = torch.tanh(
                h_t @ self.W_hh_forward + 
                x_t @ self.W_xh_forward + 
                self.b_h_forward
            )  # (batch_size, hidden_size)
            h_forward.append(h_t)

        # 反向隐藏状态序列
        h_backward = []
        h_t = h_prev_backward
        for t in range(T-1, -1, -1):
            x_t = inputs[:, t, :]  # (batch_size, input_size)
            h_t = torch.tanh(
                h_t @ self.W_hh_backward + 
                x_t @ self.W_xh_backward + 
                self.b_h_backward
            )
            h_backward.insert(0, h_t)  # 从前面插入，保证顺序对应

        # 堆叠
        h_forward = torch.stack(h_forward, dim=1)  # (batch_size, seq_len, hidden_size)
        h_backward = torch.stack(h_backward, dim=1)  # (batch_size, seq_len, hidden_size)
        h_stitched = torch.cat([h_forward, h_backward], dim=2)  # (batch_size, seq_len, hidden_size*2)

        return h_stitched