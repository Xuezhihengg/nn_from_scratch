import torch
import torch.nn as nn
import torch.nn.init as init

class RNN(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, num_layers=1, bidirectional=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 形如 self.W_xh[layer * num_directions + direction]
        self.W_xh = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.b_h = nn.ParameterList()

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                input_dim = input_size if layer == 0 else hidden_size * self.num_directions
                # 权重和偏置形状：
                # W_xh: (input_dim, hidden_size)
                # W_hh: (hidden_size, hidden_size)
                # b_h: (hidden_size,)
                self.W_xh.append(nn.Parameter(torch.empty(input_dim, hidden_size)))
                self.W_hh.append(nn.Parameter(torch.empty(hidden_size, hidden_size)))
                self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))
            
                for param in [self.W_xh[-1], self.W_hh[-1]]:
                        init.xavier_uniform_(param)




    def forward(self, inputs, h_prev=None):
        """
        inputs: (batch_size, seq_len, input_size)
        h_prev: None or tensor of shape (num_layers * num_directions, batch_size, hidden_size)
        returns:
            outputs: (batch_size, seq_len, hidden_size * num_directions)
            h_final: (num_layers * num_directions, batch_size, hidden_size)
        """

        batch_size, seq_len, _ = inputs.size()

        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=inputs.device,
            )  
        else:
            assert h_prev.size() == (
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
            )
            h_prev = h_prev.clone()  # 防止修改输入

        layer_input = inputs  # (batch_size, seq_len, input_dim)，初始input_dim = input_size
        all_final_hiddens = []  # 收集每一层每个方向的最终隐藏状态 h_t
        
        for layer in range(self.num_layers):
            outputs_layer = []
            
            # 逐方向处理序列
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                W_xh = self.W_xh[idx]  # (input_dim, hidden_size)
                W_hh = self.W_hh[idx]  # (hidden_size, hidden_size)
                b_h = self.b_h[idx]    # (hidden_size,)
                
                h_t = h_prev[idx]  # (batch_size, hidden_size)

                # 时间步顺序
                time_steps = range(seq_len) if direction == 0 else reversed(range(seq_len))

                outputs_direction = []
                for t in time_steps:
                    x_t = layer_input[:, t, :]  # (batch_size, input_dim)
                    # h_t为上一时刻隐藏状态 (batch_size, hidden_size)
                    # 计算当前时刻隐藏状态：
                    h_t = torch.tanh(x_t @ W_xh + h_t @ W_hh + b_h)  # (batch_size, hidden_size)
                    outputs_direction.append(h_t)
                
                if direction == 1:
                    # 反向序列输出反转为正向序列，以保持时间步顺序一致
                    outputs_direction.reverse()

                # 拼接时间维度，结果形状：
                # (batch_size, seq_len, hidden_size)
                outputs_direction = torch.stack(outputs_direction, dim=1)
                outputs_layer.append(outputs_direction)

                all_final_hiddens.append(h_t)  # (batch_size, hidden_size)
            
            # 双向情况下，拼接正反向隐藏状态：
            # shape: (batch_size, seq_len, hidden_size*2)
            if self.bidirectional:
                layer_output = torch.cat(outputs_layer, dim=2)
            else:
                # 单向时只有一个方向
                # (batch_size, seq_len, hidden_size)
                layer_output = outputs_layer[0]  
            
            # 下一层输入为当前层输出
            # 对于多层，上一层输出维度为 hidden_size * num_directions
            # num_directions=1时为hidden_size，=2时为hidden_size*2
            layer_input = layer_output
        
        # 将所有最终隐藏状态堆叠为 (num_layers * num_directions, batch_size, hidden_size)
        h_final = torch.stack(all_final_hiddens, dim=0)

        return layer_output, h_final


# 测试示例
if __name__ == "__main__":
    batch_size = 3
    seq_len = 5
    input_size = 10
    hidden_size = 8
    num_layers = 2
    bidirectional = True

    rnn = RNN(input_size, hidden_size, num_layers, bidirectional)

    x = torch.randn(batch_size, seq_len, input_size)
    out = rnn(x)
    print("Output shape:", out.shape)
    # 输出应为 (batch_size, seq_len, hidden_size * num_directions), 
    # 这里num_directions=2，因此为(batch_size, seq_len, 16)