import torch
import torch.nn as nn

class DeepRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100, num_layers=2):
        super(DeepRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.W_xh = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.b_h = nn.ParameterList()

        for layer in range(num_layers):
            input_dim = embedding_dim if layer == 0 else hidden_size
            self.W_xh.append(nn.Parameter(torch.randn(input_dim, hidden_size) * 0.01))
            self.W_hh.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            self.b_h.append(nn.Parameter(torch.zeros(hidden_size)))

        self.fc = nn.Linear(hidden_size, vocab_size)


    def forward(self, inputs, h_prev=None):
        """
        inputs: LongTensor of shape (batch_size, seq_length)
        h_prev: None or list/tuple of hidden states for each layer,
                each tensor shape (batch_size, hidden_size)
        """
        batch_size, T = inputs.size()

        if h_prev is None:
            h_prev = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) 
                      for _ in range(self.num_layers)]
        else:
            assert len(h_prev) == self.num_layers

        outputs = []
        h_t = h_prev  # list 每层隐藏状态张量

        for t in range(T):
            # 取第t个时间步所有batch的数据： (batch_size,)
            x_t = inputs[:, t]
            x_t = self.embedding(x_t)   # (batch_size, embedding_dim)
            
            new_h = []
            input_t = x_t  # (batch_size, input_dim)
            for layer in range(self.num_layers):
                ht_prev = h_t[layer]  # (batch_size, hidden_size)
                ht = torch.tanh(
                    input_t @ self.W_xh[layer] + 
                    ht_prev @ self.W_hh[layer] + 
                    self.b_h[layer]
                )
                new_h.append(ht)
                input_t = ht  # 传给下一层

            h_t = new_h

            y_t = self.fc(h_t[-1])  # (batch_size, vocab_size)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, T, vocab_size)

        return outputs