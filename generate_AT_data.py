import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle

# 하이퍼파라미터 및 환경
epsilon = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 1. data_mis.pkl 불러오기
with open('data_mis.pkl', 'rb') as f:
    data_mis = pickle.load(f)
mis_indices = set([item['index'] for item in data_mis])
mis_dict = {item['index']: item for item in data_mis}

# 2. 원본 train 데이터셋 불러오기
orig_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)

# 3. FGSM 함수 정의
def fgsm_attack(model, loss_fn, image, label, epsilon):
    image_adv = image.clone().detach().to(device)
    image_adv.requires_grad = True
    output = model(image_adv)
    loss = loss_fn(output, label)
    model.zero_grad()
    loss.backward()
    grad = image_adv.grad.data
    adv_image = image_adv + epsilon * grad.sign()
    # MNIST 정규화 해제 후 clipping, 다시 정규화
    mean, std = 0.1307, 0.3081
    adv_image = adv_image * std + mean
    adv_image = torch.clamp(adv_image, 0, 1)
    adv_image = (adv_image - mean) / std
    return adv_image.detach()

# 4. FGSM 생성을 위한 모델 준비 (pretrained 모델 사용)
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

pretrained_model = SimpleCNN()
if torch.cuda.device_count() > 1:
    pretrained_model = nn.DataParallel(pretrained_model)
pretrained_model = pretrained_model.to(device)
pretrained_model.load_state_dict(torch.load('model.pth', map_location=device))
pretrained_model.eval()

loss_fn = nn.CrossEntropyLoss()

# 5. 적대적 예제 포함 전체 train tensor 미리 생성 (메인 프로세스에서)
all_imgs = []
all_labels = []
for idx in range(len(orig_train_dataset)):
    img, label = orig_train_dataset[idx]
    if idx in mis_indices:
        img = img.unsqueeze(0).to(device)
        label_tensor = torch.tensor([mis_dict[idx]['true']], dtype=torch.long, device=device)
        adv_img = fgsm_attack(pretrained_model, loss_fn, img, label_tensor, epsilon)
        img = adv_img.squeeze(0).cpu()
    all_imgs.append(img)
    all_labels.append(label)
train_imgs_tensor = torch.stack(all_imgs)
train_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

# 6. 저장
torch.save(train_imgs_tensor, 'adv_train_imgs.pt')
torch.save(train_labels_tensor, 'adv_train_labels.pt')
print(f"적대적 예제 포함 train tensor 저장 완료: adv_train_imgs.pt, adv_train_labels.pt")
