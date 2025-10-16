import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 50, hidden_size = 100):
        super(SimpleLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # 初始化权重
        self.W_hi = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xi = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hf = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xf = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_ho = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xo = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_hc = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.W_xc = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.b_c = nn.Parameter(torch.zeros(self.hidden_size))
        

        # 输出层线性变换
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, h_prev = None):
        batch_size, T = inputs.size()
        if h_prev is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t, c_t = h_prev
            
        outputs = []

        for t in range(T):
            x_t = inputs[:, t]
            x_t = self.embedding(x_t)

            I_t = torch.sigmoid(h_t @ self.W_hi + x_t @ self.W_xi + self.b_i)
            F_t = torch.sigmoid(h_t @ self.W_hf + x_t @ self.W_xf + self.b_f)
            O_t = torch.sigmoid(h_t @ self.W_ho + x_t @ self.W_xo + self.b_o)

            c_tilde = torch.tanh(h_t @ self.W_hc + x_t @ self.W_xc + self.b_c)
            c_t = F_t * c_t + I_t * c_tilde

            h_t = O_t * torch.tanh(c_t)

            y_t = self.fc(h_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim = 1)    # (batch_size, T, vocab_size)


