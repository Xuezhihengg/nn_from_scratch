import logging
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from comm import Logger, Plotter

from dataset.wikitext2_dataset import WikiText2Dataset

from model.rnn import SimpleRNN, DeepRNN, BidRNN
from model.gru import SimpleGRU, DeepGRU, BidGRU
from model.lstm import SimpleLSTM, DeepLSTM, BidLSTM


# 超参数
seq_length = 10
batch_size = 16
num_epochs = 50
learning_rate = 0.001

plotter = Plotter(num_epochs)
logger = Logger(level=logging.DEBUG)

train_dataset = WikiText2Dataset(split='train', seq_length=seq_length, min_freq=2)
valid_dataset = WikiText2Dataset(split='validation', seq_length=seq_length, min_freq=2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

vocab_size = len(train_dataset.vocab)

model = BidLSTM(vocab_size)
logger.print_model_params(model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch + 1}/{num_epochs}")
    # 训练阶段
    model.train()
    total_loss_train = 0

    for i, (batch_X, batch_Y) in enumerate(train_loader):
        logger.info(f"    Batch {i}")

        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        logits = outputs[:, -1, :]

        loss = criterion(logits, batch_Y)
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
            outputs = model(batch_X)
            logits = outputs[:, -1, :]

            loss = criterion(logits, batch_Y)
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
        torch.save(model.state_dict(), os.path.join(save_path, f'BidLSTM_epoch_{epoch + 1}.pth'))
        logger.info(f"Saved model at epoch {epoch + 1}")

# 绘制训练和验证曲线
plotter.plot()