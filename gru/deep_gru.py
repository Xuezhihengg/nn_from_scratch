import torch
import torch.nn as nn


class DeepGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 50, hidden_size = 100, num_layers = 5):
        super(DeepGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 初始化权重
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
            input_dim = self.embedding_dim if layer == 0 else self.hidden_size
            self.W_hr.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xr.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_r.append(nn.Parameter(torch.zeros(self.hidden_size)))

            self.W_hz.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xz.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_z.append(nn.Parameter(torch.zeros(self.hidden_size)))

            self.W_hh.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xh.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_h.append(nn.Parameter(torch.zeros(self.hidden_size)))

        # 输出层线性变换
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, h_prev = None):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
            h_prev: tensor, shape(batch_size, hidden_size)
        """
        batch_size, T = inputs.size()
        if h_prev is None:
            h_prev = [torch.zeros(batch_size, self.hidden_size, device = inputs.device) for _ in range(self.num_layers)]
        else:
            assert len(h_prev) == self.num_layers
        
        outputs = []
        h_t = h_prev
        for t in range(T):
            # 取第t个时间步所有batch的数据： (batch_size,)
            x_t = inputs[:, t]
            x_t = self.embedding(x_t)   # (batch_size, embedding_dim)

            new_h = []
            input_t = x_t  # (batch_size, input_dim)
            for layer in range(self.num_layers):
                ht_prev = h_t[layer]

                R_t = torch.sigmoid(ht_prev @ self.W_hr[layer] + input_t @ self.W_xr[layer] + self.b_r[layer])
                Z_t = torch.sigmoid(ht_prev @ self.W_hz[layer] + input_t @ self.W_xz[layer] + self.b_z[layer])

                h_tilda = torch.tanh((R_t * ht_prev) @ self.W_hh[layer] + input_t @ self.W_xh[layer] + self.b_h[layer])
                ht = Z_t * h_tilda + (1 - Z_t) * ht_prev

                new_h.append(ht)
                input_t = ht    # 传给下一层

            h_t = new_h

            y_t = self.fc(h_t[-1])  # 通过最后一层隐藏值计算y
            outputs.append(y_t)     # len = t, 每个元素为tensor, shape(batch_size, vocab_size)

        outputs = torch.stack(outputs, dim = 1)     # (batch_size, T, vocab_size)

        return outputs