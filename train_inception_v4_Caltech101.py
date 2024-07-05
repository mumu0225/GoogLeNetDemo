import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import copy
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import timm

# 检查CUDA是否可用
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# 数据集路径
data_dir = 'Caltech 101'

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(331),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0, drop_last=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 测试数据加载
for inputs, labels in dataloaders['train']:
    print(inputs.shape)
    break

# 加载预训练的Inception V4模型
model = timm.create_model('inception_v4', pretrained=True)
num_ftrs = model.get_classifier().in_features

# 修改最后的全连接层，并添加Dropout
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(class_names))
)

# 冻结前几层
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

# 解冻更多层
layers_to_unfreeze = ['features.10', 'features.11', 'features.12', 'features.13']
for name, param in model.named_parameters():
    if any(layer in name for layer in layers_to_unfreeze):
        param.requires_grad = True

params_to_update = [p for p in model.parameters() if p.requires_grad]

# 只训练最后一层和解冻的层
optimizer = optim.AdamW(params_to_update, lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
since = time.time()

num_epochs = 50
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 20)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
        epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
        epoch_iou = jaccard_score(all_labels, all_preds, average='macro', zero_division=1)

        print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Precision: {:.4f} Recall: {:.4f} IoU: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall, epoch_iou))

        # 深度复制模型
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# 加载最佳模型权重
model.load_state_dict(best_model_wts)

# 在验证集上进行推理
model.eval()
running_corrects = 0
all_labels = []
all_preds = []

for inputs, labels in dataloaders['val']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
    all_labels.extend(labels.cpu().numpy())
    all_preds.extend(preds.cpu().numpy())

val_acc = running_corrects.double() / dataset_sizes['val']
val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
val_iou = jaccard_score(all_labels, all_preds, average='macro', zero_division=1)

print('Validation Acc: {:.4f} F1: {:.4f} Precision: {:.4f} Recall: {:.4f} IoU: {:.4f}'.format(
    val_acc, val_f1, val_precision, val_recall, val_iou))
