import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # RNN层参数,手动定义权重和偏置
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)    # (input_size, hidden_size)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)      # (hidden_size, hidden_size)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))                           # (hidden_size,)


    def forward(self, inputs, h_prev=None):
        """
        inputs: Tensor of shape (batch_size, seq_length, input_size),
        h_prev: Tensor or None, shape (batch_size, hidden_size)

        返回：
            outputs: (batch_size, seq_length, hidden_size)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入embedding维度不匹配"

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs = []
        h_t = h_prev  # (batch_size, hidden_size)

        for t in range(T):
            # 取第t个时间步所有batch的特征向量: (batch_size, input_size)
            x_t = inputs[:, t, :]
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)
            outputs.append(h_t)

        # outputs: list长度 T,每个元素 (batch_size, hidden_size)
        # 拼接时换一个维度： T 个时间步沿dim=1连接,变成 (batch_size, T, hidden_size)
        outputs = torch.stack(outputs, dim=1)

        return outputs