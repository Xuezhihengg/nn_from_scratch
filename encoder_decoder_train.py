import logging
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from comm import Logger, Plotter

from dataset.tatoeba_dataset import TatoebaDataset

from model.seq2seq_encoder_decoder import Seq2SeqEncoderDecoder


# 超参数
seq_length = 10
batch_size = 16
num_epochs = 50
learning_rate = 0.001

plotter = Plotter(num_epochs)
logger = Logger(level=logging.DEBUG)

train_dataset = TatoebaDataset("fr-en", "en", "fr", split='train')
valid_dataset = TatoebaDataset("fr-en", "en", "fr", split='valid')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

enc_vocab_size = train_dataset.source_vocab_size
dec_vocab_size = train_dataset.target_vocab_size
pad_token_id = train_dataset.pad_token_id

model = Seq2SeqEncoderDecoder(enc_vocab_size, dec_vocab_size, embedding_dim=50, hidden_size=100)
logger.print_model_params(model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch + 1}/{num_epochs}")
    # 训练阶段
    model.train()
    total_loss_train = 0

    for i, (batch_X, batch_Y) in enumerate(train_loader):
        logger.info(f"    Batch {i}")

        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)   # (batch_size, seq_length)

        dec_X = batch_Y[:, :-1]         # decoder输入: 目标序列左移一位
        target_Y = batch_Y[:, 1:]       # 标签: 目标序列右移一位

        optimizer.zero_grad()
        
        outputs = model(batch_X, dec_X)       # shape: (batch_size, seq_length-1, vocab_size)
        
        logits = outputs.reshape(-1, outputs.size(2))  # (batch_size*(seq_length-1), vocab_size)
        labels = target_Y.reshape(-1)    

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
    
    avg_loss_train = total_loss_train / len(train_loader)
    perplexity_train = math.exp(avg_loss_train)

    # 验证阶段
    model.eval()
    total_loss_valid = 0

    with torch.no_grad():
        for batch_X, batch_Y in valid_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            dec_X = batch_Y[:, :-1]        # decoder输入
            target_Y = batch_Y[:, 1:]      # 标签

            outputs = model(batch_X, dec_X)
            
            logits = outputs.reshape(-1, outputs.size(2))
            labels = target_Y.reshape(-1)                  # (batch_size * seq_length)


            loss = criterion(logits, labels)
            total_loss_valid += loss.item()

    avg_loss_valid = total_loss_valid / len(valid_loader)
    perplexity_valid = math.exp(avg_loss_valid)

    logger.info(
        f"Train Loss: {avg_loss_train:.4f}, Perplexity: {perplexity_train:.2f} | "
        f"Valid Loss: {avg_loss_valid:.4f}, Perplexity: {perplexity_valid:.2f}"
    )

    # 更新plotter数据
    plotter.update(avg_loss_train, avg_loss_valid, perplexity_train, perplexity_valid)

    # 每10个epoch保存模型
    if (epoch + 1) % 10 == 0:
        save_path = './saved_model'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, f'Seq2SeqEncoderDecoder_epoch_{epoch + 1}.pth'))
        logger.info(f"Saved model at epoch {epoch + 1}")

# 绘制训练和验证曲线
plotter.plot()