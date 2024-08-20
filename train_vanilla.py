import argparse
#import _pickle as cPickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms
import vanilla as net
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
#from torch import torchsummary

lr = 5e-5
parser = argparse.ArgumentParser()
# training options
parser.add_argument('-e', type=int, default=10,
                    help='Epochs')
parser.add_argument('-b', type=int, default=10, help='Batch Size')
parser.add_argument('-l', type=str, help='Encoder Weight File')
parser.add_argument('-s', type=str, help='Decoder Weight File')
parser.add_argument('-cuda', type=str, help='[y/N]')
args = parser.parse_args()


'''def unpickle(training_file):
    import pickle
    with open(training_file, 'rb') as fo:
        dict_image = pickle.load(fo, encoding='bytes')
    return dict_image'''


def train(n_epochs, optimizer, loss_fn, training_set,
          model_train, scheduler, device):
    losses_train = []
    print('training...')
    for epoch in range(1, n_epochs + 1):
        model_train.train()
        loss_train = 0.0
        # lrs = []
        # train_batch_accs = []
        print('epochs ', epoch)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.b, shuffle=True)
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = model_train(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        losses_train += [loss_train / len(train_loader)]
        print('Loss: ', losses_train[epoch-1])

    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(losses_train, label='train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.show()
    plt.savefig(args.loss_plot)

    # plt.show(loss_train)
    return losses_train


def adjust_learning_rate(optimizer, iteration_count, lr):
    """Imitating the original implementation"""
    learning_rate = lr / (1.0 + lr * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = net.Alexnet.decoder
    encoder = net.Alexnet.encoder
    encoder.load_state_dict(torch.load('encoder.pth'))
    encoder = nn.Sequential(*list(encoder.children())[:31])
    model = net.Alex_net(encoder=encoder, decoder=decoder)
    t_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = t_transform
    train_set = CIFAR100('./data', train=True, download=True,
                      transform=t_transform)
    test_set = CIFAR100('./data', train=False, download=True,
                     transform=test_transform)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.decoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    '''optimizer = optim.SGD(params=model.decoder.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)'''

    train(n_epochs=args.e, optimizer=optimizer,
          loss_fn=loss_fn, training_set=train_set, model_train=model,
          scheduler=scheduler, device=device)
    torch.save(model.state_dict(), args.s)


if __name__ == '__main__':
    main()

    # fine_labels in dictionary for image labels in training data set
    # dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
