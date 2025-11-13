import torch
import torch.nn as nn
import torch.nn.init as init


class GRU(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, num_layers=1, bidirectional=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.W_hr = nn.ParameterList()
        self.W_xr = nn.ParameterList()
        self.b_r = nn.ParameterList()

        self.W_hz = nn.ParameterList()
        self.W_xz = nn.ParameterList()
        self.b_z = nn.ParameterList()

        self.W_hh = nn.ParameterList()
        self.W_xh = nn.ParameterList()
        self.b_h = nn.ParameterList()

        # 初始化所有参数
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                input_dim = input_size if layer == 0 else hidden_size * self.num_directions

                # 重置门
                self.W_hr.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xr.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_r.append(nn.Parameter(torch.zeros(hidden_size)))

                # 更新门
                self.W_hz.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xz.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_z.append(nn.Parameter(torch.zeros(hidden_size)))

                # 候选隐藏状态
                self.W_hh.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.W_xh.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))

                # Xavier 初始化
                for param in [self.W_hr[-1], self.W_xr[-1], self.W_hz[-1], self.W_xz[-1], self.W_hh[-1], self.W_xh[-1]]:
                    init.xavier_uniform_(param)

    def forward(self, inputs, h_prev=None):
        """
        Args:
            inputs: (batch_size, seq_len, input_size)
            h_prev: None or (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            outputs: (batch_size, seq_len, hidden_size * num_directions)
            h_final: (num_layers * num_directions, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # 初始化隐藏状态
        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size,
                device=device
            )
        else:
            assert h_prev.size() == (
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )
            h_prev = h_prev.clone()

        layer_input = inputs
        all_final_hiddens = []

        for layer in range(self.num_layers):
            outputs_layer = []

            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction

                W_hr = self.W_hr[idx]
                W_xr = self.W_xr[idx]
                b_r = self.b_r[idx]

                W_hz = self.W_hz[idx]
                W_xz = self.W_xz[idx]
                b_z = self.b_z[idx]

                W_hh = self.W_hh[idx]
                W_xh = self.W_xh[idx]
                b_h = self.b_h[idx]

                h_t = h_prev[idx]

                time_steps = range(seq_len) if direction == 0 else reversed(range(seq_len))

                outputs_direction = []
                for t in time_steps:
                    x_t = layer_input[:, t, :]  # (batch_size, input_dim)

                    R_t = torch.sigmoid(x_t @ W_xr + h_t @ W_hr + b_r)
                    Z_t = torch.sigmoid(x_t @ W_xz + h_t @ W_hz + b_z)
                    h_tilda = torch.tanh(x_t @ W_xh + (R_t * h_t) @ W_hh + b_h)
                    h_t = Z_t * h_tilda + (1 - Z_t) * h_t 
                    outputs_direction.append(h_t)

                if direction == 1:
                    outputs_direction.reverse()

                outputs_direction = torch.stack(outputs_direction, dim=1)  # (batch_size, seq_len, hidden_size)
                outputs_layer.append(outputs_direction)

                all_final_hiddens.append(h_t)

            if self.bidirectional:
                layer_output = torch.cat(outputs_layer, dim=2)  # (batch_size, seq_len, 2*hidden_size)
            else:
                layer_output = outputs_layer[0]

            layer_input = layer_output

        h_final = torch.stack(all_final_hiddens, dim=0) # (num_layers * num_directions, batch_size, hidden_size)

        return layer_output, h_final