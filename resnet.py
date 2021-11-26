from torchvision import models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from runner import train
from dataloader import get_data
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

clf = nn.Sequential(OrderedDict([('Fully_Connected_Layer_1', nn.Linear(2048, 256)),
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(256, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))


model.fc = clf
print(model.fc)
opt = optim.Adam(model.fc.parameters(), lr=0.003)
criterion = nn.NLLLoss()
train_loader, test_loader = get_data('transformed_data', 'transformed_data')
print("============================== DATA LOADED ==============================") 
train(model, opt, criterion, train_loader, test_loader, 'Resnet', 'LMS-less', 1, 10)
