import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        # d_model 是输入向量的维度大小，即embedding的维度
        # max_len 是最大序列长度
        # pe 是位置编码矩阵, 尺寸为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1) 形如[[0], [1], [2], ..., [max_len-1]]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 分母项, div_term_k = exp((-ln(10000)/d_model)*k), k为偶数下标（0, 2, 4,...)   div_term.shape = (d_model/2,)

        # 偶数位置使用正弦
        # 广播机制自动扩展形状: position.shape = (max_len, 1), div_term.shape = (d_model/2,)  
        # 广播后乘积 shape = (max_len, d_model/2)  
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数位置使用余弦
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展维度方便批次广播, 1 对应batch维度，方便后续和输入张量 (batch_size, seq_len, d_model) 相加
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不作为参数训练，但保存到模型

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)     # 这里不是d_model->d_k多次,而是先d_model->d_model,再拆分多头,即用一个Linear变换把输入直接映射成所有头拼接的维度(d_model),再拆成多个头
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def scaled_dot_product_attention(self, q, k, v, mask = None):
        # q, k, v: (batch_size, num_heads, seq_len, d_k)
        # output: (batch, heads, seq_len_q, d_k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            # 遮挡位置赋值为很小的负数，使softmax趋近于0       
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim = -1)
        attn = self.dropout(attn)
        output = attn @ v   # (batch, heads, seq_len_q, d_k)
        return output, attn
    
    def forward(self, query, key ,value, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        # out: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        # 线性变换 + 多头拆分
        # (batch_size, seq_len, d_model)->(batch_size, seq_len, d_model)->(batch_size, seq_len, num_heads, d_k)->(batch, num_heads, seq_len, d_k)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)     # 一定要这样吗？
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        out,attn = self.scaled_dot_product_attention(q, k, v, mask)

        # 拼接所有heads
        # (batch_size, num_heads, seq_len, d_k)->(batch_size, seq_len, num_heads, d_k)->(batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)

        return out, attn
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, d_model)
        return self.net(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # x: (batch_size, seq_len, d_model)
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, input_vocab_size, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # src: (batch_size, src_len,)
        x = self.embedding(src) * math.sqrt(self.d_model) # (batch_size, src_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        # tgt: (batch_size, tgt_len,)
        x = self.embedding(tgt) * math.sqrt(self.d_model)     # (batch_size, tgt_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, input_vocab_size, max_len, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len, dropout)
        self.out_linear = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # 屏蔽padding
        # (batch_size, src_len,) -> (batch_size, 1, src_len) -> (batch_size, 1, 1, src_len)
        return (src !=0).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        # 屏蔽padding
        # (batch_size, tgt_len,) -> (batch_size, 1, tgt_len) -> (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) 
        tgt_len = tgt.size(1)
        # future mask
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device = tgt.device)).bool()     # (tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # 广播机制 (batch_size, 1, tgt_len, tgt_len)
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        out = self.out_linear(dec_out)      # (batch_size, tgt_len, tgt_vocab_size)
        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    src_len = 5
    tgt_len = 4
    input_vocab_size = 10
    tgt_vocab_size = 8

    # src: 有效token随机生成1~9，0为padding
    src = torch.tensor([
        [1, 7, 0, 0, 0],
        [3, 2, 4, 5, 0]
    ], dtype=torch.long)

    # tgt: 注意tgt的输入通常为“previous tokens”，与输出错开一个位置
    tgt = torch.tensor([
        [1, 2, 0, 0],
        [3, 4, 5, 0]
    ], dtype=torch.long)

    # 初始化Transformer
    transformer = Transformer(input_vocab_size, tgt_vocab_size)

    # 前向运行
    out = transformer(src, tgt)  # (batch, tgt_len, tgt_vocab_size)

    print("Output shape:", out.shape)  # 期望 (2, 4, 8)
    print("Output example (batch 0, first token logits):", out[0, 0])
    
    # 也可以验证下mask
    src_mask = transformer.make_src_mask(src)
    tgt_mask = transformer.make_tgt_mask(tgt)
    print("src_mask shape:", src_mask.shape)  # (2, 1, 1, 5)
    print("src_mask example (batch 0):", src_mask[0])
    print("tgt_mask shape:", tgt_mask.shape)  # (2, 1, 4, 4)
    print("tgt_mask example (batch 0):", tgt_mask[0][0])

    





















        




