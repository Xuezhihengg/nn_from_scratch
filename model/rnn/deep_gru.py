import torch
import torch.nn as nn

class DeepGRU(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, num_layers=5):
        super(DeepGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_hr = nn.ParameterList()
        self.W_xr = nn.ParameterList()
        self.b_r = nn.ParameterList()

        self.W_hz = nn.ParameterList()
        self.W_xz = nn.ParameterList()
        self.b_z = nn.ParameterList()

        self.W_hh = nn.ParameterList()
        self.W_xh = nn.ParameterList()
        self.b_h = nn.ParameterList()

        for layer in range(self.num_layers):
            input_dim = input_size if layer == 0 else hidden_size
            self.W_hr.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xr.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_r.append(nn.Parameter(torch.zeros(hidden_size)))

            self.W_hz.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xz.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_z.append(nn.Parameter(torch.zeros(hidden_size)))

            self.W_hh.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xh.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))

    def forward(self, inputs, h_prev=None):
        """
        Args:
            inputs: Tensor, shape (batch_size, seq_len, input_size)
            h_prev: None or list of hidden states, length=num_layers,
                each tensor shape (batch_size, hidden_size)
        Returns:
            outputs: Tensor, shape (batch_size, seq_len, hidden_size),
                最后一层隐藏状态序列
        """
        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev is None:
            h_prev = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        else:
            assert len(h_prev) == self.num_layers

        h_t = h_prev
        outputs = []
        for t in range(T):
            input_t = inputs[:, t, :]  # (batch_size, input_size)
            new_h = []

            for layer in range(self.num_layers):
                ht_prev = h_t[layer]

                R_t = torch.sigmoid(ht_prev @ self.W_hr[layer] + input_t @ self.W_xr[layer] + self.b_r[layer])
                Z_t = torch.sigmoid(ht_prev @ self.W_hz[layer] + input_t @ self.W_xz[layer] + self.b_z[layer])

                h_tilda = torch.tanh((R_t * ht_prev) @ self.W_hh[layer] + input_t @ self.W_xh[layer] + self.b_h[layer])
                ht = Z_t * h_tilda + (1 - Z_t) * ht_prev

                new_h.append(ht)
                input_t = ht  # 传给下一层

            h_t = new_h
            outputs.append(h_t[-1])  # 记录最后一层隐藏状态

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        return outputs