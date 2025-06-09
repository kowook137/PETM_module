import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 구조 (SimpleCNN)
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

# 데이터셋 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

# 모델 불러오기 함수
def load_model(weight_path):
    model = SimpleCNN()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 정확도 평가 함수
def evaluate(model, loader):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            all_preds.append(pred.cpu())
            all_labels.append(target.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = correct / total
    return acc, all_preds, all_labels

if __name__ == "__main__":
    # 모델 불러오기
    model_std = load_model("model.pth")
    model_at = load_model("AT_model.pth")

    # 정확도 및 예측값
    acc_std, preds_std, labels = evaluate(model_std, train_loader)
    acc_at, preds_at, _ = evaluate(model_at, train_loader)

    print(f"[model.pth] Train Accuracy: {acc_std*100:.2f}%")
    print(f"[AT_model.pth] Train Accuracy: {acc_at*100:.2f}%")

    # 예측 불일치 분석
    diff_mask = preds_std != preds_at
    low_accuracy_mask = diff_mask
    std_wrong_mask = preds_std != labels

    # low_accuracy 중 model.pth가 틀린 비율
    low_acc_and_wrong = (low_accuracy_mask & std_wrong_mask).sum().item()
    low_acc_total = low_accuracy_mask.sum().item()
    if low_acc_total > 0:
        low_acc_wrong_ratio = low_acc_and_wrong / low_acc_total
    else:
        low_acc_wrong_ratio = 0.0

    # model.pth가 틀린 데이터 중 low_accuracy인 비율
    std_wrong_total = std_wrong_mask.sum().item()
    std_wrong_and_low_acc = (std_wrong_mask & low_accuracy_mask).sum().item()
    if std_wrong_total > 0:
        std_wrong_low_acc_ratio = std_wrong_and_low_acc / std_wrong_total
    else:
        std_wrong_low_acc_ratio = 0.0

    print(f"Among low_accuracy (prediction mismatch), model.pth wrong ratio: {low_acc_wrong_ratio*100:.2f}%")
    print(f"Among model.pth wrong predictions, low_accuracy ratio: {std_wrong_low_acc_ratio*100:.2f}%")
