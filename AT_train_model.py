import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# 하이퍼파라미터 (pretrain_model.py와 동일)
epochs = 5
batch_size = 128
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 적대적 예제 포함 train tensor 불러오기
train_imgs_tensor = torch.load('adv_train_imgs.pt')
train_labels_tensor = torch.load('adv_train_labels.pt')

at_train_dataset = TensorDataset(train_imgs_tensor, train_labels_tensor)
at_train_loader = DataLoader(at_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# test_loader 준비 (원본과 동일)
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 정의 (pretrain_model.py와 동일)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data, target in at_train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[AT] Epoch {epoch}, Loss: {total_loss / len(at_train_loader):.4f}")

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    print(f"[AT] Test Accuracy: {100. * correct / total:.2f}%")

if __name__ == "__main__":
    train()
    test()
    torch.save(model.state_dict(), "AT_model.pth")
    print("AT_model.pth 저장 완료.")
