# Code-transmission

## code_anim.py
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pygame
import sys
import numpy as np

# ---------------------- 1. 数据与模型部分 ----------------------
# 构建文本数据（单词级，可替换为字符级）
texts = [
    "hello world",
    "hello yellow",
    "coding is fun",
    "python is cool",
    "animation vs code"
]

# 构建词汇表
vocab = set()
for text in texts:
    vocab.update(text.split())
vocab = sorted(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

# 文本数据集类（预测下一个词）
class TextDataset(Dataset):
    def __init__(self, texts, word_to_idx, seq_length=2):
        self.texts = texts
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length
        self.data = []
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - seq_length):
                x = tokens[i:i+seq_length]
                y = tokens[i+seq_length]
                self.data.append((x, y))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_tokens, y_token = self.data[idx]
        x = torch.tensor([self.word_to_idx[tok] for tok in x_tokens], dtype=torch.long)
        y = torch.tensor(self.word_to_idx[y_token], dtype=torch.long)
        return x, y

# LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=10, hidden_dim=20, seq_length=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.seq_length = seq_length
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

# 初始化数据集、模型、损失函数、优化器
dataset = TextDataset(texts, word_to_idx, seq_length=2)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = LSTMModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 预测下一个词的函数
def predict_next_word(input_sequence, model, word_to_idx, idx_to_word):
    input_tokens = input_sequence.split()
    input_idx = torch.tensor([word_to_idx[tok] for tok in input_tokens], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(input_idx)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]

# ---------------------- 2. Pygame 可视化部分（整合动画交互） ----------------------
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Animation vs. Coding")
font = pygame.font.SysFont("Consolas", 24)
input_text = ""
predicted_word = ""

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # 按下回车时，模型预测下一个词
                if input_text.strip():
                    predicted_word = predict_next_word(input_text, model, word_to_idx, idx_to_word)
            elif event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            else:
                input_text += event.unicode

    screen.fill((0, 0, 0))
    # 绘制输入提示和文本
    input_surface = font.render(f"Input: {input_text}", True, (255, 255, 255))
    screen.blit(input_surface, (50, 100))
    # 绘制模型预测结果
    if predicted_word:
        result_surface = font.render(f"Predicted: {predicted_word}", True, (255, 255, 255))
        screen.blit(result_surface, (50, 150))
    pygame.display.flip()
```
