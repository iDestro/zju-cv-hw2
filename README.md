# Handwriting digits recognize
浙江大学软件学院计算机视觉导论第二次作业

### 1 Introduction to MNIST dataset

MNIST dataset come from a National Institute of standards and technology(NIST).Training set compose of handwriting digits from 250 different people,half of them are high school student, others are worker from the Census Bureau.Test set also have the same proportion with Training set.

Here are some examples of training pictures:

![](C:\Users\Administrator\PycharmProjects\zju-cv-hw2\pictures\samples_of_training_set.png)

### 2 LeNet

Network architecture:

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [1, 6, 30, 30]              60
              ReLU-2             [1, 6, 30, 30]               0
         MaxPool2d-3             [1, 6, 15, 15]               0
            Conv2d-4            [1, 16, 11, 11]           2,416
              ReLU-5            [1, 16, 11, 11]               0
         MaxPool2d-6              [1, 16, 5, 5]               0
            Linear-7                   [1, 120]          48,120
       BatchNorm1d-8                   [1, 120]             240
              ReLU-9                   [1, 120]               0
           Linear-10                    [1, 84]          10,164
      BatchNorm1d-11                    [1, 84]             168
             ReLU-12                    [1, 84]               0
           Linear-13                    [1, 10]             850
================================================================
Total params: 62,018
Trainable params: 62,018
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.13
Params size (MB): 0.24
Estimated Total Size (MB): 0.37
----------------------------------------------------------------
```

Implement with Pytorch:

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
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
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

### 3 Training & evaluate

command:

```bash
python run.py
```

result:

```bash
Test Loss:  0.04,  Test Acc: 99.15%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

           0     0.9919    0.9959    0.9939       980
           1     0.9947    0.9974    0.9960      1135
           2     0.9894    0.9932    0.9913      1032
           3     0.9912    0.9980    0.9946      1010
           4     0.9859    0.9959    0.9909       982
           5     0.9910    0.9899    0.9905       892
           6     0.9948    0.9896    0.9922       958
           7     0.9912    0.9844    0.9878      1028
           8     0.9948    0.9887    0.9918       974
           9     0.9900    0.9812    0.9856      1009

    accuracy                         0.9915     10000
   macro avg     0.9915    0.9914    0.9914     10000
weighted avg     0.9915    0.9915    0.9915     10000

Confusion Matrix...
[[ 976    0    1    0    0    0    2    1    0    0]
 [   0 1132    0    0    0    1    0    2    0    0]
 [   1    1 1025    0    1    0    1    3    0    0]
 [   0    1    0 1008    0    1    0    0    0    0]
 [   0    0    0    0  978    1    1    0    0    2]
 [   2    0    0    4    0  883    1    1    0    1]
 [   4    2    0    0    2    2  948    0    0    0]
 [   0    2    6    2    1    0    0 1012    1    4]
 [   0    0    4    3    0    0    0    1  963    3]
 [   1    0    0    0   10    3    0    1    4  990]]
```