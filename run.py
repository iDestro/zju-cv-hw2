import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Config, LeNet
from train_eval import train
from utils import visualize_network_architecture


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
    train_iter = DataLoader(train_set,
                            shuffle=True,
                            batch_size=config.batch_size)

    test_iter = DataLoader(test_set,
                           shuffle=True,
                           batch_size=config.batch_size)

    model = LeNet()
    visualize_network_architecture(model)
    train(model, config, train_iter, test_iter)
