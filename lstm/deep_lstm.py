import torch
import torch.nn as nn

class DeepLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 50, hidden_size = 100, num_layers = 5):
        super(DeepLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 初始化权重
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
            input_dim = self.embedding_dim if layer == 0 else self.hidden_size

            self.W_hi.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xi.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_i.append(nn.Parameter(torch.zeros(self.hidden_size)))

            self.W_hf.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xf.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_f.append(nn.Parameter(torch.zeros(self.hidden_size)))

            self.W_ho.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xo.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_o.append(nn.Parameter(torch.zeros(self.hidden_size)))

            self.W_hc.append(nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01))
            self.W_xc.append(nn.Parameter(torch.randn(input_dim, self.hidden_size) * 0.01))
            self.b_c.append(nn.Parameter(torch.zeros(self.hidden_size)))
            

        # 输出层线性变换
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, h_prev = None):
        batch_size, T = inputs.size()
        if h_prev is None:
            h_t = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        else:
            assert len(h_prev) == self.num_layers
            h_t, c_t = h_prev
            
        outputs = []

        for t in range(T):
            x_t = self.embedding(inputs[:, t])

            new_h = []
            new_c = []
            input_t = x_t

            for layer in range(self.num_layers):
                ht_prev = h_t[layer]
                ct_prev = c_t[layer]

                I_t = torch.sigmoid(ht_prev @ self.W_hi[layer] + input_t @ self.W_xi[layer] + self.b_i[layer])
                F_t = torch.sigmoid(ht_prev @ self.W_hf[layer] + input_t @ self.W_xf[layer] + self.b_f[layer])
                O_t = torch.sigmoid(ht_prev @ self.W_ho[layer] + input_t @ self.W_xo[layer] + self.b_o[layer])

                c_tilde = torch.tanh(ht_prev @ self.W_hc[layer] + input_t @ self.W_xc[layer] + self.b_c[layer])
                ct = F_t * ct_prev + I_t * c_tilde

                ht = O_t * torch.tanh(ct)

                new_h.append(ht)
                new_c.append(ct)

                input_t = ht

            h_t = new_h
            c_t = new_c

            y_t = self.fc(h_t[-1])
            outputs.append(y_t)

        return torch.stack(outputs, dim = 1)    # (batch_size, T, vocab_size)


