import torch
import torch.nn as nn


class DeepRNN(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, num_layers=2):
        super(DeepRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_xh = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.b_h = nn.ParameterList()

        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else hidden_size
            self.W_xh.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.W_hh.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))

    def forward(self, inputs, h_prev=None):
        """
        inputs: Tensor of shape (batch_size, seq_length, input_size)
        h_prev: None or list of hidden states for each layer,
                each tensor shape (batch_size, hidden_size)
        Returns:
            outputs: Tensor of shape (batch_size, seq_length, hidden_size)
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev is None:
            h_prev = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) 
                      for _ in range(self.num_layers)]
        else:
            assert len(h_prev) == self.num_layers

        outputs = []
        h_t = h_prev  # 当前每层隐藏状态列表

        for t in range(T):
            input_t = inputs[:, t, :]  # 输入第t时间步 (batch_size, input_size)
            new_h = []
            for layer in range(self.num_layers):
                ht_prev = h_t[layer]
                ht = torch.tanh(
                    input_t @ self.W_xh[layer] + 
                    ht_prev @ self.W_hh[layer] + 
                    self.b_h[layer]
                )
                new_h.append(ht)
                input_t = ht  # 传给下一层

            h_t = new_h
            outputs.append(h_t[-1])  # 记录最后一层隐藏状态

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return outputs