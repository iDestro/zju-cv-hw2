import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Config, LeNet
from train_eval import train

train_set = torchvision.datasets.MNIST('./data',
                                       train=True,
                                       transform=transforms.ToTensor())
test_set = torchvision.datasets.MNIST('./data',
                                      train=False,
                                      transform=transforms.ToTensor())
train_iter = DataLoader(train_set,
                        shuffle=True,
                        batch_size=128)

test_iter = DataLoader(test_set,
                       shuffle=True,
                       batch_size=128)

config = Config()
model = LeNet()

train(model, config, train_iter, test_iter)
