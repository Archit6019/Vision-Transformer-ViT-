import numpy as np
import  torch
import glob
import os
import yaml
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm 
from typing import List, Any
import zipfile
from sklearn.model_selection import train_test_split
from VIT import ViT

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --Config Parsing
def parse_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = parse_config('config.yaml')
print("Config file loaded")

# --Dataset Class
class Test_Dataset(Dataset):
    def __init__(self, file_list, transform = None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)

        label = img_path.split("\\")[-2]
        label = 1 if label == "African" else 0

        return img_transformed, label
    
# --Transforms (Image transformation)
train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),]
)
test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),]
)

# --VIT Class 
model = ViT(
    dim= config['VIT']['dim'],
    image_size= config['VIT']['image_size'],
    patch_size = config['VIT']['patch_size'],
    num_classes = config['VIT']['num_classes'],
    depth= config['VIT']['depth'],
    heads= config['VIT']['heads'],
    mlp_dim = config['VIT']['mlp_dim'],
    channels = config['VIT']['channels']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=float(config['VIT']['lr']))
scheduler = StepLR(optimizer, step_size=1, gamma=config['VIT']['gamma'])

# --Train Class (main)
class Train_Class:
    def __init__(self, dir : str,  sub1 : List[str], sub2 :List[str]):
        self.dir = dir
        self.sub1 = sub1
        self.sub2 = sub2 
        self.train_list = []
        self.test_list = []
        self.valid_list = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def preprocess(self):
        for sub in self.sub1:
            for sub_ in self.sub2:
                path = os.path.join(self.dir, sub, sub_)
                if os.path.isdir(path):
                    try:
                        jpg_files = glob.glob(os.path.join(path, "*.jpg"))
                        if sub == 'train':
                            self.train_list.extend(jpg_files)
                        elif sub == 'test':
                            self.test_list.extend(jpg_files)
                        else:
                            raise ValueError("Unexpected value for sub")
                    except Exception as e:
                        print(f"Error processing directory {path}: {e}")
                else:
                    print(f"Directory not found at {path}")

        labels = [path.split('\\')[-2] for path in self.train_list]
        self.train_list , self.valid_list = train_test_split(self.train_list,
                                           test_size=0.2,
                                           stratify=labels)
        
        train_data = Test_Dataset(self.train_list, transform=train_transforms)
        valid_data = Test_Dataset(self.valid_list, transform=test_transforms)
        test_data = Test_Dataset(self.test_list, transform=test_transforms)

        self.train_loader = DataLoader(dataset=train_data, batch_size=config['VIT']['batch_size'], shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_data, batch_size=config['VIT']['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=config['VIT']['batch_size'], shuffle=True)
        print("Preprocessing complete, model ready for training")

    def train(self):
        best_val_accuracy = 0
        print("Beginning training for VIT")
        for epoch in range(config['VIT']['epochs']):
            EPOCH_LOSS = 0
            EPOCH_ACCURACY = 0

            for data, label in self.train_loader:
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                EPOCH_ACCURACY += acc / len(self.train_loader)
                EPOCH_LOSS += loss / len(self.train_loader)

            with torch.no_grad():
                EPOCH_VAL_ACCURACY = 0
                EPOCH_VAL_LOSS = 0
                for data , label in self.valid_loader:
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    val_acc = (val_output.argmax(dim=1) == label).float().mean()
                    EPOCH_VAL_ACCURACY += val_acc / len(self.valid_loader)
                    EPOCH_VAL_LOSS += val_loss / len(self.valid_loader)

            print(
            f"Epoch : {epoch+1} - loss : {EPOCH_LOSS:.4f} - acc: {EPOCH_ACCURACY:.4f} - val_loss : {EPOCH_VAL_LOSS:.4f} - val_acc: {EPOCH_VAL_ACCURACY:.4f}\n"
        )
            
            if EPOCH_VAL_ACCURACY > best_val_accuracy:
                best_val_accuracy = EPOCH_VAL_ACCURACY
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")
            
if __name__ == '__main__':
    ins = Train_Class(dir= "dataset", sub1=['train', 'test'], sub2=['African', 'Asian'])
    ins.preprocess()
    ins.train()
