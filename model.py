import torch.nn as nn
import torch
from tensorboardX import SummaryWriter


class Config:
    def __init__(self):
        self.model_name = 'MNIST'
        self.learn_rate = 0.02
        self.num_epochs = 20
        self.batch_size = 1024
        self.dropout = 0
        self.reduce = 1
        self.class_list = [str(i) for i in range(10)]
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6//config.reduce, 3, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6//config.reduce, 16//config.reduce, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Sequential(nn.Linear(16//config.reduce * 5 * 5, 120),
                                 nn.BatchNorm1d(120),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84),
                                 nn.BatchNorm1d(84),
                                 nn.ReLU(),
                                 nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    dummy_input = torch.rand(13, 1, 28, 28)  # 假设输入13张1*28*28的图片
    model = LeNet(config=Config())
    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(model, dummy_input)
