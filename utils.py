from model import LeNet
from torchsummary import summary


def visualize_network_architecture(model):
    summary(model.cuda(), (1, 28, 28), batch_size=1)
