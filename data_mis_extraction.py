import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

# pretrain_model.py와 동일한 모델 구조
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

# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 전처리 및 로더 (pretrain_model.py와 동일)
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)

# 모델 생성 및 파라미터 로드
model = SimpleCNN()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# 오답 데이터 선별
data_mis = []  # (image tensor, true label, pred label, index)
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)
        mis_mask = preds != target
        if mis_mask.any():
            mis_data = data[mis_mask].cpu()
            mis_target = target[mis_mask].cpu()
            mis_pred = preds[mis_mask].cpu()
            mis_indices = torch.arange(batch_idx * train_loader.batch_size, batch_idx * train_loader.batch_size + data.size(0))[mis_mask.cpu()]
            for img, t, p, idx in zip(mis_data, mis_target, mis_pred, mis_indices):
                data_mis.append({'image': img, 'true': int(t), 'pred': int(p), 'index': int(idx)})

# data_mis 저장 (pickle)
with open('data_mis.pkl', 'wb') as f:
    pickle.dump(data_mis, f)

print(f"wrong prediction data {len(data_mis)} number saved in data_mis.pkl")
