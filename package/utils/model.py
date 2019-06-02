import torch.nn as nn


def init_classifier(m):
    if isinstance(m, nn.Linear): #判断对象是否是已知的类型 m是否是nn.Linear
        nn.init.normal_(m.weight, std=0.001) #初始化权重根据标准差
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) #初始化偏差

#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
def create_embedding(in_dim=None, out_dim=None):
    layers = [
        nn.Linear(in_dim, out_dim), #线性变换
        nn.BatchNorm1d(out_dim), #批量标准化
        nn.ReLU(inplace=True) #正则化
    ]
    return nn.Sequential(*layers)
#<class 'list'>: [Linear(in_features=2048, out_features=512, bias=True),
#  BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True,
#  track_running_stats=True), ReLU(inplace)]