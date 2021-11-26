from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data(train_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(160), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(160), transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir + '/Train', transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir + '/Test', transform=test_transforms)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader =  DataLoader(test_data, batch_size=128, shuffle=False)

    return train_loader, test_loader