
import re
import torch
from collections import Counter
from torch.utils.data import Dataset
from datasets import load_dataset
from comm import Logger

class WikiText2Dataset(Dataset):
    def __init__(self, split='train', seq_length=30, min_freq=2, logger = None):
        """
        完整封装WikiText2加载、预处理、构建词表及数字化，生成训练样本。

        参数:
            split (str): 'train', 'validation'或'test'
            seq_length (int): 输入序列长度
            min_freq (int): 词汇表最小词频，低于此频率词映射为 <unk>

        使用示例:
            dataset = WikiText2Dataset(split='train', seq_length=30)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        """
        super().__init__()

        if logger is None:
            self.logger = Logger("dataset.WikiText2Dataset")
        
        # 1. 载入和预处理文本
        self.tokens = self._load_and_tokenize(split)
        
        # 2. 构建词表
        self.vocab, self.word2id, self.id2word = self._build_vocab(self.tokens, min_freq)
        
        # 3. tokens转id
        self.data_ids = self._tokens_to_ids(self.tokens)

        # 4. 构造样本序列长度
        self.seq_length = seq_length
        self.n_samples = len(self.data_ids) - self.seq_length
        assert self.n_samples > 0, "数据长度必须大于seq_length"


    def preprocess_line(self, line):
        """
        清洗文本: 小写，替换非字母数字+标点为空格，多空格合一，去除首尾空格
        """
        line = line.lower()
        line = re.sub(r"[^a-z0-9\s.,;!?'\-]", ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        return line


    def _load_and_tokenize(self, split):
        """
        加载数据集并预处理分词
        """
        hf_split = 'validation' if split == 'valid' else split
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=hf_split)
        tokens = []
        for example in dataset:
            text = example['text'] # type: ignore
            if text.strip():
                line = self.preprocess_line(text)
                if line:
                    tokens.extend(line.split(' '))
        tokens = [tok for tok in tokens if tok]  # 过滤空串
        self.logger.info(f"<{split}> Tokens count: {len(tokens)}, Unique tokens: {len(set(tokens))}")
        return tokens


    def _build_vocab(self, tokens, min_freq):
        """
        构建词表,低频词归类到 <unk>
        """
        counter = Counter(tokens)
        vocab = [word for word, freq in counter.items() if freq >= min_freq]

        # 特殊token优先放置
        vocab = ['<pad>', '<unk>'] + sorted(vocab)

        word2id = {word: idx for idx, word in enumerate(vocab)}
        id2word = {idx: word for word, idx in word2id.items()}
        self.logger.info(f"Vocabulary size: {len(vocab)}")
        return vocab, word2id, id2word


    def _tokens_to_ids(self, tokens):
        unk_id = self.word2id.get('<unk>')
        ids = [self.word2id.get(tok, unk_id) for tok in tokens]
        return ids


    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):
        x = self.data_ids[idx: idx + self.seq_length]
        y = self.data_ids[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)