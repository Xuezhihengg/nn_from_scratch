import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100):
        super(SimpleRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN层参数，手动定义权重和偏置
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)    # (embedding_dim, hidden_size)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)      # (hidden_size, hidden_size)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))                           # (hidden_size,)

        # 输出层线性变换
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, inputs, h_prev=None):
        """
        inputs: LongTensor of shape (batch_size, seq_length)
        h_prev: Tensor or None, shape (batch_size, hidden_size)
        """
        batch_size, T = inputs.size()

        if h_prev is None:
            # 初始化隐藏状态为0
            h_prev = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs = []
        h_t = h_prev  # (batch_size, hidden_size)

        for t in range(T):
            # 取第t个时间步所有batch的数据： (batch_size,)
            x_t = inputs[:, t]
            # Embedding层直接支持batch输入，输出 (batch_size, embedding_dim)
            x_t = self.embedding(x_t)

            # x_t @ W_xh: (batch_size, embedding_dim) @ (embedding_dim, hidden_size) = (batch_size, hidden_size)
            # h_t @ W_hh: (batch_size, hidden_size) @ (hidden_size, hidden_size) = (batch_size, hidden_size)
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)

            y_t = self.fc(h_t)   # (batch_size, vocab_size)
            outputs.append(y_t)

        # outputs: list长度 T，每个元素 (batch_size, vocab_size)
        # 拼接时换一个维度： T 个时间步沿dim=1连接，变成 (batch_size, T, vocab_size)
        outputs = torch.stack(outputs, dim=1)

        return outputs