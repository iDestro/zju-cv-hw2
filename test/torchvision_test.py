import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train = torchvision.datasets.MNIST('../data',
                                   train=True,
                                   transform=transforms.ToTensor())
test = torchvision.datasets.MNIST('../data',
                                  train=False,
                                  transform=transforms.ToTensor())
train_dataloader = DataLoader(train,
                              shuffle=True,
                              batch_size=128)

test_dataloader = DataLoader(test,
                             shuffle=True,
                             batch_size=128)


img, label = next(iter(test_dataloader))
plt.imshow(img)
plt.show()