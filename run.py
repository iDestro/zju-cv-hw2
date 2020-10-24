import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Config, LeNet
from train_eval import train
from utils import visualize_network_architecture
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--flag', '-f', action='store_true', help='Is the number of convolution kernels halved?')
parser.add_argument('--dropout_rate', '-d', type=float, default=0.0, help='Please set a dropout rate')
args = parser.parse_args()

if __name__ == '__main__':

    train_set = torchvision.datasets.MNIST('./',
                                           train=True,
                                           download=True,
                                           transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST('./',
                                          train=False,
                                          download=True,
                                          transform=transforms.ToTensor())
    config = Config()
    print(args.flag)
    config.reduce = 2 if args.flag else 1
    config.dropout = args.dropout_rate
    print(config.reduce)
    print(config.dropout)
    train_iter = DataLoader(train_set,
                            shuffle=True,
                            batch_size=config.batch_size)

    test_iter = DataLoader(test_set,
                           shuffle=True,
                           batch_size=config.batch_size)

    model = LeNet(config)
    visualize_network_architecture(model)
    train(model, config, train_iter, test_iter)
