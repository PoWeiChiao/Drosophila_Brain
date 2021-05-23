import cv2 as cv
import glob
import os
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from model.ResUNet import BasicBlock, ResUNet
from model.UNet import UNet

def predict(net, device, image_path, image_transforms, threshold=0.5):
    image = Image.open(image_path)
    output_image = np.array(image, dtype=np.uint8)
    image = image_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        pred = net(image)
        pred = np.array(pred.data.cpu()[0])[0]
        pred = np.where(pred >= threshold, 255, 0)
        pred = np.array(pred, dtype=np.uint8)
        return output_image, pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1)
    # net = ResUNet(in_channel=1, out_channel=1, block=BasicBlock, num_block=[3, 4, 6, 3])
    net.to(device=device)
    net.load_state_dict(torch.load('model.pth', map_location=device))

    data_dir = 'data/test'
    image_path = os.path.join(data_dir, 'image', '2.png')
    image_transformers = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    image, pred = predict(net, device, image_path, image_transformers)

if __name__ == '__main__':
    main()