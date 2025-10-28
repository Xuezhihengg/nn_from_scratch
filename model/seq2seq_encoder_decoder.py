import torch
import torch.nn as nn
from .rnn import SimpleGRU

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100):
        super(Seq2SeqEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = SimpleGRU(self.embedding_dim, self.hidden_size)
    
    def forward(self, inputs):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
        Return:
            outputs: tensor, shape(batch_size, seq_length, hidden_size)
        """
        inputs = self.embedding(inputs)     # (batch_size, seq_length, embedding_dim)
        return self.rnn(inputs)
    

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=100):
        super(Seq2SeqDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = SimpleGRU(self.embedding_dim + self.hidden_size, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, state):
        """
        Args:
            inputs: tensor, shape(batch_size, seq_length)
            state: tensor, shape(batch_size, hidden_size)
        Return:
            outputs: tensot, shape(batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = inputs.size()
        inputs = self.embedding(inputs)     # (batch_size, seq_length, embedding_dim)
        context = state.unsqueeze(1).repeat(1, seq_length, 1)   # (batch_size, seq_length, hidden_size)
        inputs_and_context = torch.cat([inputs, context], dim=2)    # (batch_size, seq_length, embedding_dim+hidden_size)
        outputs = self.rnn(inputs_and_context)  # (batch_size, seq_length, hidden_size)
        return self.dense(outputs)   # (batch_size, seq_length, vocab_size)
    

    def forward_step(self, input_t, state):
        """
        Args:
            input_t: (batch_size,) 当前步输入token id
            state: (batch_size, hidden_size) 当前hidden状态
        Return:
            output: (batch_size, vocab_size) 当前步词分布
            next_state: (batch_size, hidden_size) 下一步hidden状态
        """
        batch_size = input_t.size(0)
        embedded = self.embedding(inpuat_t)          # (batch_size, embedding_dim)
        context = state                             # (batch_size, hidden_size)
        inputs = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (batch_size, 1, embedding_dim+hidden_size)
        output, next_state = self.rnn.forward_step(inputs, state)   # output: (batch_size, 1, hidden_size)
        output = output.squeeze(1)                  # (batch_size, hidden_size)
        output = self.dense(output)                 # (batch_size, vocab_size)
        return output, next_state
    

    def 
        


class Seq2SeqEncoderDecoder(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size ,embedding_dim=50, hidden_size=100):
        super(Seq2SeqEncoderDecoder, self).__init__()
        self.encoder = Seq2SeqEncoder(enc_vocab_size, embedding_dim, hidden_size)
        self.decoder = Seq2SeqDecoder(dec_vocab_size, embedding_dim, hidden_size)

    def forward(self, enc_X, dec_X):
        """
        Args:
            enc_X: 源语言输入, tensor, shape(batch_size, seq_length)
            dec_X: 目标语言输入, tensor, shape(batch_size, seq_length)
        Return:
            outputs: tensot, shape(batch_size, seq_length, vocab_size)
        """
        enc_outputs = self.encoder(enc_X)   # (batch_size, seq_length, hidden_size)
        dec_state = enc_outputs[:, -1, :]   # (batch_size, hidden_size)
        return self.decoder(dec_X, dec_state)   









