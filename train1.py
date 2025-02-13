import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import SqueezeNet
from dataloader import LoadData
import copy

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SqueezeNet(num_classes=2).to(device)

train_data = LoadData("/Users/micoria/Desktop/SqueezeNet/train.txt", train_flag=True)
print(f"✅ TDataset size: {len(train_data)}")
train_dl = DataLoader(train_data, batch_size=32, shuffle=True)

valid_data = LoadData("/Users/micoria/Desktop/SqueezeNet/valid.txt", train_flag=False)
print(f"✅ VDataset size: {len(valid_data)}")
valid_dl = DataLoader(valid_data, batch_size=32, shuffle=False)

# ✅ **more stable optimizer + flexible learning rate**
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # ✅ AdamW 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # ✅ 每 10 轮降低 lr

# 🔥 train parameters
epochs = 50
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []

plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()
        total += y.size(0)

    train_losses.append(total_loss / len(train_dl))
    train_accs.append(correct / total)

    # 🔍 calculate loss and accuracy
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for X_val, y_val in valid_dl:
            X_val, y_val = X_val.to(device), y_val.to(device)
            output_val = model(X_val)
            val_loss += loss_function(output_val, y_val).item()
            val_correct += (output_val.argmax(1) == y_val).sum().item()
            val_total += y_val.size(0)

    valid_losses.append(val_loss / len(valid_dl))
    valid_accs.append(val_correct / val_total)

    # 🎨 update visualization
    ax[0].cla()
    ax[0].plot(range(1, epoch+2), train_losses, label="Train Loss", marker='o')
    ax[0].plot(range(1, epoch+2), valid_losses, label="Valid Loss", marker='s')
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    
    ax[1].cla()
    ax[1].plot(range(1, epoch+2), train_accs, label="Train Accuracy", marker='o')
    ax[1].plot(range(1, epoch+2), valid_accs, label="Valid Accuracy", marker='s')
    ax[1].set_title("Accuracy Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    
    plt.pause(0.5)

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Train Acc = {train_accs[-1]*100:.2f}% | Valid Loss = {valid_losses[-1]:.4f}, Valid Acc = {valid_accs[-1]*100:.2f}% | LR = {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), "/Users/micoria/Desktop/SqueezeNet/best_model.pth")

    scheduler.step()  # ✅ update learning rate

plt.ioff()
plt.show()
