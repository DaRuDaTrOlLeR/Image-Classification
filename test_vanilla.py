import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import vanilla as net
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-l', type=str, help='Encoder Weight File')
parser.add_argument('-s', type=str, help='Decoder Weight File')
parser.add_argument('-cuda', type=str, help='[y/N]')
args = parser.parse_args()

def test(model, test_set, device):
    correct_1 = 0.0
    correct_5 = 0.0
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True)
    for imgs, labels in test_loader:
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load('decoder.pth'))
            outputs = model(imgs)
            _, predicted = outputs.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(predicted)
            correct = predicted.eq(labels).float()
            correct_1 += correct[:, :1].sum()
            correct_5 += correct[:, :5].sum()

    print('Top1 Error: %.3f%% | Top5 Error: %.3f%%'
          % (100 * (1 - correct_1 / len(test_loader.dataset)), 100 * (1 - correct_5 / len(test_loader.dataset))))

def main():
	device = torch.device("cuda" if torch.cuda)
