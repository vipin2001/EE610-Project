from torchvision import models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from runner import train
from dataloader import get_data
model = models.efficientnet_b1(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
print(model.classifier)
clf = nn.Sequential(OrderedDict([('Dropout_1', nn.Dropout(p=0.2, inplace=True)),
                                 ('Fully_Connected_Layer_1', nn.Linear(1280, 512)),
                                 ('ReLU_1', nn.ReLU()),
                                 ('Fully_Connected_Layer_2', nn.Linear(512, 256)),
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(256, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))


model.classifier = clf
print(model.classifier)
opt = optim.Adam(model.classifier.parameters(), lr=0.003)
criterion = nn.NLLLoss()
train_loader, test_loader = get_data('original_data', 'original_data')
print("============================== DATA LOADED ==============================")
train(model, opt, criterion, train_loader, test_loader, 'EfficientNet', 'RGB', 1, 10)