import torch
import numpy as np
from torch.utils.data import Dataset , DataLoader
from torchvision import datasets , transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import time
from datetime import datetime

from MyOptimizer import MySGD

import wandb


# 检测可用设备
device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Apple MPS")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("nvidia CUDA")
else:
    print("CPU")

################################################################################################################################
################################################################################################################################
#构建模型
class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.MyLayers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1 , out_channels = 16 , kernel_size = (3 , 3)),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels = 16 , out_channels = 32 , kernel_size = (3 , 3)),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 73728 , out_features = 3),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 3 , out_features = 3),
            torch.nn.Softmax()
        )

    def forward(self , x):
        return self.MyLayers(x)

################################################################################################################################
################################################################################################################################

#训练和验证
def MyTrain(model , TrainDataLoader , TestDataLoader , EpochNum , optimizer , LossFunction , SavePath):
    
    run = wandb.init(
    #设置项目名称
    project="MyCNN",
    #设计runs记录的名称（这里用时间）
    name = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #设置runs记录的描述
    config={
        "optimizer": optimizer,
    },
)



    MinLoss = -1
    for epoch in range(EpochNum):
        print('\nepoch', epoch + 1 , end='\n')
        # 模型训练
        model.train()  # 开始训练
        for BatchIndex, (x , y) in enumerate(TrainDataLoader):
            x, y = x.to(device), y.to(device)
            result = model(x)
            loss = LossFunction(result.to(torch.float32), y.to(torch.long))

            loss.backward()  # 根据loss计算梯度
            optimizer.step()  # 根据梯度调整模型
            optimizer.zero_grad()  # 梯度归零

            predict = torch.max(input = result, dim = 1)[1]  # 将预测的最大值作为预测结果，torch.max()[0]返回最大值，torch.max()[1]返回最大值下标
            correct = torch.eq(predict , y).sum().item()
            accuracy = correct / TrainDataLoader.batch_size

            print('\r', end='')
            print(BatchIndex + 1, ' / ' , len(TrainDataLoader) , '  acc:', accuracy , '; loss:' , loss.item() , end='' , sep='')    

        #模型测试
        model.eval()#开启推理
        print('\n' , end = '')
        with torch.no_grad():   #关闭模型参数修改
            ValLoss = 0
            ValAccuracy = 0
            num = 0
            for BatchIndex , (x, y) in enumerate(TestDataLoader):
                x, y = x.to(device), y.to(device)
                result = model(x)
                ValLoss += LossFunction(result.to(torch.float32), y.to(torch.long)).item()

                predict = torch.max(input=result, dim=1)[1]
                correct = torch.eq(predict , y).sum().item()
                ValAccuracy += correct / TestDataLoader.batch_size

                num += 1
            print('\r', end='')
            ValAccuracy /= num
            ValLoss /= num
            print('ValAcc:' , ValAccuracy , '; ValLoss:' , ValLoss , end='' , sep='')

            

            if ValLoss < MinLoss or MinLoss == -1:
                print('\nValLoss:', MinLoss, '->', ValLoss)
                MinLoss = ValLoss
                torch.save(model , SavePath)

        run.log({'accuracy' : accuracy , "loss": loss.item()})
        run.log({'ValAccuracy' : ValAccuracy , "ValLoss": ValLoss})




transform = transforms.Compose([
    transforms.Resize([200 , 200]),
    transforms.Grayscale(),
    transforms.ToTensor()
])

TrainDataset = ImageFolder(root = 'train' , transform = transform)
TrainLoader = DataLoader(dataset = TrainDataset , batch_size = 100 , shuffle = True , drop_last = True)
#数据集：dataset；每批次的大小：8；是否打乱：True；最后不足一批是否舍弃：True
TestDataset = ImageFolder(root = 'test' , transform = transform)
TestLoader = DataLoader(dataset = TestDataset , batch_size = 10 , shuffle = True , drop_last = True)

MyModel = CNNModel().to(device)

ModelName = 'model' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.pth'
MyTrain(model = MyModel,
        TrainDataLoader = TrainLoader,
        TestDataLoader = TestLoader,
        EpochNum = 1000,
        # optimizer = torch.optim.SGD(params = MyModel.parameters() , lr = 0.0005),
        optimizer = MySGD(params = MyModel.parameters() , lr = 0.0005),
        LossFunction = torch.nn.CrossEntropyLoss(),
        SavePath = ModelName
        )