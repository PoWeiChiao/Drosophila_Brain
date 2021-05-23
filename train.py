import os
from tqdm import tqdm
from model.DeepLab_ResNet import DeepLabv3_plus
from model.ResUNet import BasicBlock, BottleNeck, ResUNet
from model.UNet import UNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from utils.dataset import DrosophilaDataset
from utils.DiceLoss import DiceLoss
from utils.logger import Logger

def train(net, device, dataset, batch_size=1, epochs=50, lr=0.00001):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(params=net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceLoss()

    best_loss = float('inf')
    log = Logger('log_train.txt')

    for epoch in range(epochs):
        loss_train = 0.0
        print('running epoch: {}'.format(epoch))
        net.train()
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_train += loss.item() * image.size(0)

            loss.backward()
            optimizer.step()

        loss_train = loss_train / len(train_loader.dataset)
        print('\tTraining Loss: {:.6f}'.format(loss_train))
        if loss_train < best_loss:
            best_loss = loss_train    
            torch.save(net.state_dict(), 'model.pth')
            print('model saved')
        log.write_line(str(epoch) + ',' + str(round(loss_train, 6)))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ' + str(device))
    
    data_dir = 'data/train'
    image_dir = os.path.join(data_dir, 'image')
    label_dir = os.path.join(data_dir, 'label')
    image_transformers = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = DrosophilaDataset(image_dir, label_dir, image_transformers)

    net = DeepLabv3_plus(in_channels=1, n_classes=1, os=16)
    # net = UNet(n_channels=1, n_classes=1)
    # net = ResUNet(in_channel=1, out_channel=1, block=BasicBlock, num_block=[3, 4, 6, 3])
    net.to(device=device)

    train(net, device, dataset, batch_size=2)

if __name__ =='__main__':
    main()