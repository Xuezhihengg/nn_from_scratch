import torch
import torch.nn as nn

class DeepLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, num_layers=5):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_hi = nn.ParameterList()
        self.W_xi = nn.ParameterList()
        self.b_i = nn.ParameterList()

        self.W_hf = nn.ParameterList()
        self.W_xf = nn.ParameterList()
        self.b_f = nn.ParameterList()

        self.W_ho = nn.ParameterList()
        self.W_xo = nn.ParameterList()
        self.b_o = nn.ParameterList()

        self.W_hc = nn.ParameterList()
        self.W_xc = nn.ParameterList()
        self.b_c = nn.ParameterList()

        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else hidden_size

            self.W_hi.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xi.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_i.append(nn.Parameter(torch.zeros(hidden_size)))

            self.W_hf.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xf.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_f.append(nn.Parameter(torch.zeros(hidden_size)))

            self.W_ho.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xo.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_o.append(nn.Parameter(torch.zeros(hidden_size)))

            self.W_hc.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.W_xc.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.b_c.append(nn.Parameter(torch.zeros(hidden_size)))

    def forward(self, inputs, h_prev=None):
        """
        inputs: Tensor, shape (batch_size, seq_len, input_size)
        h_prev: tuple of lists (h_t_list, c_t_list), or None
            h_t_list and c_t_list: each list of num_layers tensors with shape (batch_size, hidden_size)
        Returns:
            outputs: Tensor, shape (batch_size, seq_len, hidden_size)
        """

        batch_size, T, input_size = inputs.size()
        assert input_size == self.input_size, "输入维度不匹配"

        if h_prev is None:
            h_t = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = h_prev
            assert len(h_t) == self.num_layers and len(c_t) == self.num_layers

        outputs = []

        for t in range(T):
            input_t = inputs[:, t, :]  # (batch_size, input_size)
            new_h = []
            new_c = []

            for layer in range(self.num_layers):
                ht_prev = h_t[layer]
                ct_prev = c_t[layer]

                I_t = torch.sigmoid(ht_prev @ self.W_hi[layer] + input_t @ self.W_xi[layer] + self.b_i[layer])
                F_t = torch.sigmoid(ht_prev @ self.W_hf[layer] + input_t @ self.W_xf[layer] + self.b_f[layer])
                O_t = torch.sigmoid(ht_prev @ self.W_ho[layer] + input_t @ self.W_xo[layer] + self.b_o[layer])

                C_tilde = torch.tanh(ht_prev @ self.W_hc[layer] + input_t @ self.W_xc[layer] + self.b_c[layer])
                ct = F_t * ct_prev + I_t * C_tilde

                ht = O_t * torch.tanh(ct)

                new_h.append(ht)
                new_c.append(ct)

                input_t = ht  # 下一层输入为当前层输出

            h_t = new_h
            c_t = new_c

            outputs.append(h_t[-1])  # 记录最后一层的隐藏状态

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        return outputs