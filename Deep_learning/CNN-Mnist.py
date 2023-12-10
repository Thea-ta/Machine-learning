import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


EPOCH = 3
BATCH_SIZE=60
LR = 0.001 # 学习率
DOWNLOAD_MNIST = False #下载好以后改成false
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),   #把rgb层0-255压缩成(0,1)
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data = torchvision.datasets.MNIST(root = './mnist/',train = False) #提取的是testdata
test_loader = Data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential( #(1,28,28)图片维度、长、宽
            nn.Conv2d(#卷积层-过滤器，多少个filter,就有多少个种类
                in_channels=1,#图片有多少层rgb有三层，灰度图就两层
                out_channels=16, #提取出了多少个特征，放到下一层
                kernel_size=5, # mnist里的图片一个大小是28*28，用5*5大小的区域不断扫描
                stride=1, # 每隔多少步跳一下
                padding=2, # 扫描到边界会有多出来的，给他周围多加一层为0的数据(呈黑色)
                # if stride=1,padding=(kernel_size-1)/2
            ),  # 变成了(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #已经生成了更厚的图片，取一个最大的区域，筛选重要的部分
            #因为只选取了一个点，减小了一倍 -->(16,14,14)
        )
        self.conv2 = nn.Sequential( #-->(16,14,14)
            nn.Conv2d(16,32,5,1,2), #-->(32,14,14)
            nn.ReLU(), # 激活 #-->(32,14,14)
            nn.MaxPool2d(2),# 再次筛选-->(32,7,7)
        )
        self.out = nn.Linear(32*7*7,10) #要把三维的展平

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        x = x.view(x.size(0),-1) # -1的过程就是把维度的数据全变到一起 
                                #(batch,32*7*7)
        output = self.out(x)
        return output
    

def test(_test_loader, _model, _device):
    loss_list = []
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss = 0
    correct = 0
    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        for step1,(data, target) in enumerate(_test_loader):
            data, target = data.to(_device), target.to(_device)
            output = cnn(data.reshape(-1, 1, 28, 28))
            loss += loss_func(output, target).item()  # 添加损失值
            if step1 %100 ==0:
                loss_list.append(loss_func(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]  # 找到概率最大的下标，为输出值
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # .cpu()是将参数迁移到cpu上来。
 
    loss /= len(_test_loader.dataset)
 
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
train_accs = []

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader): 
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()  
        if step %100 ==0:
            print('Epoch:',epoch,'| train loss:%.4f' % loss.item())
    test(train_loader,cnn, DEVICE)
