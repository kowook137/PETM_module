import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 멀티 GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터
epochs = 5
batch_size = 128
learning_rate = 0.001

# 데이터 전처리 및 로더
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 간단한 CNN 모델 정의
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

# 멀티 GPU 사용
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} Using GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)}")
    model = nn.DataParallel(model)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 학습 함수
def train():
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

# 평가 및 예측 함수
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
    print(f"Test Accuracy: {100. * correct / total:.2f}%")

# 메인 실행
if __name__ == "__main__":
    train()
    test()
    # 학습된 모델 저장
    torch.save(model.state_dict(), "model.pth")
    # 예측 예시 (테스트셋 첫 10개)
    model.eval()
    with torch.no_grad():
        sample_data, _ = next(iter(test_loader))
        sample_data = sample_data.to(device)
        output = model(sample_data[:10])
        preds = output.argmax(dim=1)
        print("Predictin result for 10 samples:", preds.cpu().numpy())
