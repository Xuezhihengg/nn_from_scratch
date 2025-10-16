import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import get_tokens, build_vocab, tokens_to_ids, create_dataset
from utils import print_model_params

from rnn import SimpleRNN, DeepRNN, BidRNN
from gru import SimpleGRU, DeepGRU, BidGRU
from lstm import SimpleLSTM, DeepLSTM, BidLSTM


# 参数设置
seq_length = 10
batch_size = 16
num_epochs = 50
learning_rate = 0.001


# 1. 准备数据
tokens_train = get_tokens('train')
tokens_valid = get_tokens('validation')

vocab, word2id, id2word = build_vocab(tokens_train)

data_train = tokens_to_ids(tokens_train, word2id)
data_valid = tokens_to_ids(tokens_valid, word2id)

X_train, Y_train = create_dataset(data_train, seq_length)    # X: (num_train_samples, seq_length), Y: (num_train_samples,)
X_valid, Y_valid = create_dataset(data_valid, seq_length)    # X: (num_valid_samples, seq_length), Y: (num_valid_samples,)

vocab_size = len(vocab)
model = BidLSTM(vocab_size)
print_model_params(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# 训练过程记录
loss_list_train = []
loss_list_valid = []
perplexity_list_train = []
perplexity_list_valid = []


for epoch in range(num_epochs):
    # 训练模式
    model.train()
    total_loss_train = 0
    correct_train = 0
    total_train = 0


    print("epoch_count: ", epoch)

    for i in range(0, len(X_train) - batch_size, batch_size):
        print("     batch_count: ", i)

        batch_X = torch.tensor(X_train[i:i + batch_size], dtype=torch.long)  # (batch_size, seq_length)
        batch_Y = torch.tensor(Y_train[i:i + batch_size], dtype=torch.long)  # (batch_size, )

        optimizer.zero_grad()
        outputs = model.forward(batch_X)   # (batch_size, seq_length, vocab_size)
        logits = outputs[:, -1, :]

        # 取每条序列最后一个时间步的输出作为该序列的预测，(batch_size, vocab_size)
        loss = criterion(logits, batch_Y)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        _, predicted = torch.max(logits, dim=1)
        correct_train += (predicted == batch_Y).sum().item()
        total_train += batch_Y.size(0)

    avg_loss_train = total_loss_train / (len(X_train) // batch_size)
    perplexity_train = math.exp(avg_loss_train)

    loss_list_train.append(avg_loss_train)
    perplexity_list_train.append(perplexity_train)


    # 验证模式
    model.eval()
    total_loss_valid = 0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for i in range(0, len(X_valid) - batch_size, batch_size):
            batch_X = torch.tensor(X_valid[i:i + batch_size], dtype=torch.long)
            batch_Y = torch.tensor(Y_valid[i:i + batch_size], dtype=torch.long)

            outputs = model.forward(batch_X)
            logits = outputs[:, -1, :]

            loss = criterion(logits, batch_Y)
            total_loss_valid += loss.item()

            _, predicted = torch.max(logits, dim=1)
            correct_valid += (predicted == batch_Y).sum().item()
            total_valid += batch_Y.size(0)

    avg_loss_valid = total_loss_valid / (len(X_valid) // batch_size)
    perplexity_valid = math.exp(avg_loss_valid)

    loss_list_valid.append(avg_loss_valid)
    perplexity_list_valid.append(perplexity_valid)

    print(
        f"Epoch[{epoch + 1}/{num_epochs}] "
        f"Train Loss: {avg_loss_train:.4f}, Train Perplexity: {perplexity_train:.2f}% | "
        f"Valid Loss: {avg_loss_valid:.4f}, Valid Perplexity: {perplexity_valid:.2f}%"
    )

    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        save_path = './saved_model'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, f'simple_rnn_epoch_{epoch + 1}.pth'))
        print(f"--> Saved model at epoch {epoch + 1}")


# 绘制训练与验证的Loss和Accuracy曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_list_train, label='Train Loss')
plt.plot(range(1, num_epochs + 1), loss_list_valid, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), perplexity_list_train, label='Train Perplexity')
plt.plot(range(1, num_epochs + 1), perplexity_list_valid, label='Valid Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./training_validation_curves.png')
print("Training and validation curves saved to ./training_validation_curves.png")
plt.show()