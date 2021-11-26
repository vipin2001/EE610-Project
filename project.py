import torch
import torch.nn as nn
from torchvision import transforms, models
import argparse, textwrap
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
from collections import OrderedDict
import warnings
from colorama import init
from termcolor import cprint 
import sys
from pyfiglet import figlet_format
warnings.filterwarnings("ignore")



def RGB2LMS(img_file_path, if_convert=False):
    image = Image.open(img_file_path)
    if if_convert:
        rgb = np.asarray(image)
        rgb2xyz = np.array([[0.4124,0.3756,0.1805],[0.2126,0.7152,0.0722],[0.0192,0.1192,0.9505]])
        xyz2xyzd = np.array([[0.1884,0.6597,0.1016],[0.2318,0.8116,-0.0290],[0,0,1]])
        xyz2rgb = np.linalg.inv(rgb2xyz)
        M1 = np.matmul(xyz2rgb,xyz2xyzd)
        M2 = np.matmul(M1,rgb2xyz)
        rgb_reshaped = rgb.reshape((rgb.shape[0]*rgb.shape[1]),3)
        rgbd_reshaped = np.matmul(rgb_reshaped,M2.T).astype(int)
        rgbd = rgbd_reshaped.reshape(rgb.shape)
        rgbd_final = np.clip(rgbd,0,255)
        rgbd_image = Image.fromarray(rgbd_final.astype(np.uint8))
        img_name = img_file_path.split("\\")[-1].split(".")[0]
        rgbd_image.save("temp" + '/' + img_name+ "_transformed" + ".jpg")
        return rgbd_image
    else:
        img_name = img_file_path.split("\\")[-1].split(".")[0]
        shutil.copy(img_file_path, "temp/" + img_name + "_transformed" + ".jpg")
        return image


def image_loader(loader, image_name, if_convert):
    image = RGB2LMS(image_name, if_convert)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="EE 610 Project", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", help="Takes in path to input image file", type=str, metavar="i")
    parser.add_argument("--convert", help="Converts image to canine vision space (default is True)", type=bool, metavar="c", default=True)
    parser.add_argument("--model", help=textwrap.dedent('''Enter either of
        [1] d   : DenseNet
        [2] r   : ResNet
        [3] e   : EfficientNet'''), type=str, metavar="m", default='d')
    parser.add_argument("--mode", help="Choose if you want the Lp mode or Sp mode. Default is Lp", metavar="md", type=str, default="Lp")
    args = parser.parse_args()
    Path("temp").mkdir(exist_ok=True)
 
    if args.model == 'd': 
        saved_model = models.densenet121(pretrained=True)
        if args.mode == 'Sp':
            clf = nn.Sequential(OrderedDict([('Fully_Connected_Layer_1', nn.Linear(1024, 128)),
                                            ('ReLU_3', nn.ReLU()),
                                            ('Fully_Connected_Layer_3', nn.Linear(128, 2)),
                                            ('Output', nn.LogSoftmax(dim=1))]))

            saved_model.classifier = clf
            saved_model.load_state_dict(torch.load('DenseNet-LMS-less.pth')['state_dict']())
            
        elif args.mode == 'Lp':
            clf = nn.Sequential(OrderedDict([('Fully_Connected_Layer_1', nn.Linear(1024, 512))                  ,('ReLU_1', nn.ReLU()),
                                 ('Fully_Connected_Layer_2', nn.Linear(512, 256)),
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(256, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))


            saved_model.classifier = clf
            saved_model.load_state_dict(torch.load('DenseNet-LMS.pth')['state_dict']())
        
        saved_model.eval()

    elif args.model == 'r': 
        saved_model = models.resnet50(pretrained=True)
        if args.mode == 'Sp':
            clf = nn.Sequential(OrderedDict([('Fully_Connected_Layer_1', nn.Linear(2048, 256)),
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(256, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))

            saved_model.classifier = clf
            saved_model.load_state_dict(torch.load('Resnet-LMS-less.pth')['state_dict']())
            
        elif args.mode == 'Lp':
            clf = nn.Sequential(OrderedDict([('Fully_Connected_Layer_1', nn.Linear(2048, 512)),
                                 ('ReLU_1', nn.ReLU()),
                                 ('Fully_Connected_Layer_2', nn.Linear(512, 256)),
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(256, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))


            saved_model.fc = clf
            saved_model.load_state_dict(torch.load('Resnet-LMS.pth')['state_dict']())
        
        saved_model.eval()
    
    elif args.model == 'e': 
        saved_model = models.efficientnet_b1(pretrained=True)
        if args.mode == 'Sp':
            clf = nn.Sequential(OrderedDict([('Dropout_1', nn.Dropout(p=0.2, inplace=True)),
                                 ('Fully_Connected_Layer_1', nn.Linear(1280, 128)),
                                
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(128, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))

            saved_model.classifier = clf
            saved_model.load_state_dict(torch.load('EfficientNet-LMS-less.pth')['state_dict']())
            
        elif args.mode == 'Lp':
            clf = nn.Sequential(OrderedDict([('Dropout_1', nn.Dropout(p=0.2, inplace=True)),
                                 ('Fully_Connected_Layer_1', nn.Linear(1280, 512)),
                                 ('ReLU_1', nn.ReLU()),
                                 ('Fully_Connected_Layer_2', nn.Linear(512, 256)),
                                 ('ReLU_3', nn.ReLU()),
                                 ('Fully_Connected_Layer_3', nn.Linear(256, 2)),
                                 ('Output', nn.LogSoftmax(dim=1))]))


            saved_model.classifier = clf
            saved_model.load_state_dict(torch.load('EfficientNet-LMS.pth')['state_dict']())
        
        saved_model.eval()



    data_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])



    classes = {0: 'Cat', 1: 'Dog'}
    result = np.argmax(saved_model(image_loader(data_transforms, args.image, args.convert)).detach().numpy())

    cprint(figlet_format(classes[result], font='starwars'),
       'yellow', 'on_red', attrs=['bold'])

        
