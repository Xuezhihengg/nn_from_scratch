import torch
import torch.nn as nn
import torch.nn.init as init


class LSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

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
            for direction in range(self.num_directions):
                input_dim = input_size if layer == 0 else hidden_size * self.num_directions

                # 输入门
                self.W_hi.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xi.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_i.append(nn.Parameter(torch.zeros(hidden_size)))

                # 遗忘门
                self.W_hf.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xf.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_f.append(nn.Parameter(torch.zeros(hidden_size)))

                # 输出门
                self.W_ho.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xo.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_o.append(nn.Parameter(torch.zeros(hidden_size)))

                # 候选细胞状态
                self.W_hc.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xc.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_c.append(nn.Parameter(torch.zeros(hidden_size)))

                # 使用 Xavier 初始化（更稳定）
                for param in [self.W_hi[-1], self.W_xi[-1],
                              self.W_hf[-1], self.W_xf[-1],
                              self.W_ho[-1], self.W_xo[-1],
                              self.W_hc[-1], self.W_xc[-1]]:
                    init.xavier_uniform_(param)

    def forward(self, inputs, h_prev=None, c_prev=None):
        """
        Args:
            inputs: (batch_size, seq_len, input_size)
            h_prev: None or (num_layers * num_directions, batch_size, hidden_size)
            c_prev: None or (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            outputs: (batch_size, seq_len, hidden_size * num_directions)
            (h_final, c_final): each shape (num_layers * num_directions, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size,
                device=device, requires_grad=False
            )
        else:
            assert h_prev.size() == (
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )
            h_prev = h_prev.clone()

        if c_prev is None:
            c_prev = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size,
                device=device, requires_grad=False
            )
        else:
            assert c_prev.size() == (
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )
            c_prev = c_prev.clone()

        layer_input = inputs
        all_final_hiddens = []  
        all_final_cells = []    

        for layer in range(self.num_layers):
            outputs_layer = []

            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction

                W_hi, W_xi, b_i = self.W_hi[idx], self.W_xi[idx], self.b_i[idx]
                W_hf, W_xf, b_f = self.W_hf[idx], self.W_xf[idx], self.b_f[idx]
                W_ho, W_xo, b_o = self.W_ho[idx], self.W_xo[idx], self.b_o[idx]
                W_hc, W_xc, b_c = self.W_hc[idx], self.W_xc[idx], self.b_c[idx]

                h_t = h_prev[idx]  # (batch_size, hidden_size)
                c_t = c_prev[idx]  # (batch_size, hidden_size)

                time_steps = range(seq_len) if direction == 0 else reversed(range(seq_len))

                outputs_direction = []
                for t in time_steps:
                    x_t = layer_input[:, t, :]  # (batch_size, input_dim)

                    i_t = torch.sigmoid(x_t @ W_xi + h_t @ W_hi + b_i)
                    f_t = torch.sigmoid(x_t @ W_xf + h_t @ W_hf + b_f)
                    o_t = torch.sigmoid(x_t @ W_xo + h_t @ W_ho + b_o)
                    g_t = torch.tanh(x_t @ W_xc + h_t @ W_hc + b_c)

                    c_t = f_t * c_t + i_t * g_t
                    h_t = o_t * torch.tanh(c_t)

                    outputs_direction.append(h_t)

                if direction == 1:
                    outputs_direction.reverse()

                outputs_direction = torch.stack(outputs_direction, dim=1)  # (batch_size, seq_len, hidden_size)
                outputs_layer.append(outputs_direction)

                all_final_hiddens.append(h_t)
                all_final_cells.append(c_t)

            if self.bidirectional:
                layer_output = torch.cat(outputs_layer, dim=2)  # (batch_size, seq_len, 2*hidden_size)
            else:
                layer_output = outputs_layer[0]

            layer_input = layer_output

        h_final = torch.stack(all_final_hiddens, dim=0)  # (num_layers * num_directions, batch_size, hidden_size)
        c_final = torch.stack(all_final_cells, dim=0)

        return layer_output, (h_final, c_final)