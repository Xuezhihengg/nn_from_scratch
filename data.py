import re
from collections import Counter
from datasets import load_dataset


def preprocess_line(line):
    """
    简单文本清洗:小写,剔除非字母数字及指定标点符号,用空格替换特殊符号
    """
    line = line.lower()
    line = re.sub(r"[^a-z0-9\s.,;!?'\-]", ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def get_tokens(split='train'):
    """
    使用Huggingface datasets加载WikiText2指定split数据，清洗后分词
    split可选'train','validation','test'
    """
    # Huggingface的WikiText2的split名称是 'train', 'validation', 'test'
    hf_split = 'validation' if split == 'valid' else split

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=hf_split)
    tokens = []
    for example in dataset:
        text = example['text']
        if text.strip():
            line = preprocess_line(text)
            if line:
                tokens.extend(line.split(' '))

    tokens = [tok for tok in tokens if tok]
    print(f"Tokens count for {split}: {len(tokens)}, Unique tokens: {len(set(tokens))}")
    return tokens

def build_vocab(tokens, min_freq=2):
    """
    根据tokens构建词汇表,频率低于min_freq的词统一用<unk>
    返回:vocab列表,word2id,id2word
    """
    counter = Counter(tokens)
    # 过滤低频词
    vocab = [word for word, freq in counter.items() if freq >= min_freq]

    # 增加特殊词
    vocab = ['<pad>', '<unk>'] + sorted(vocab)

    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for word, idx in word2id.items()}

    print(f"Vocabulary size: {len(vocab)}")
    return vocab, word2id, id2word


def tokens_to_ids(tokens, word2id):
    """
    将token列表转换成对应数字id列表,未登录词映射为<unk> id
    """
    unk_id = word2id.get('<unk>')
    ids = [word2id.get(token, unk_id) for token in tokens]
    return ids


def create_dataset(data, seq_length):
    """
    构造输入序列X和目标序列Y,例如:
    X: [w0, w1, ..., w(n-1)], Y: [w1, w2, ..., w(n)]
    返回两个列表:X (n_samples * seq_length), Y (n_samples)
    """
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length])
    return X, Y