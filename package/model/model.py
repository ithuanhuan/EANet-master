from __future__ import print_function
from itertools import chain
import torch
import torch.nn as nn
from .base_model import BaseModel
from .backbone import create_backbone
from .pa_pool import PAPool
from .pcb_pool import PCBPool
from .global_pool import GlobalPool
from .ps_head import PartSegHead
from ..utils.model import create_embedding
from ..utils.model import init_classifier


class Model(BaseModel):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone)
        self.pool = eval('{}(cfg)'.format(cfg.pool_type))
        self.create_em_list()
        if hasattr(cfg, 'num_classes') and cfg.num_classes > 0:#cfg.num_classes = 751
            self.create_cls_list()
        if cfg.use_ps:
            cfg.ps_head.in_c = self.backbone.out_c
            self.ps_head = PartSegHead(cfg.ps_head)
        print('Model Structure:\n{}'.format(self))
    # self.em_list =
    # ModuleList(
    #     (0): Sequential(
    #     (0): Linear(in_features=2048, out_features=512, bias=True)
    # (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (2): ReLU(inplace)
    # )
    # )
#ModuleList是Module的子类，所有 nn.ModuleList 内部的 nn.Module 的 parameter 也被添加作为 我们的网络的 parameter
    def create_em_list(self):
        cfg = self.cfg #cfg为backbone 网络结构 resnet50的配置信息 em_dim = 512 num_classes = 751
        self.em_list = nn.ModuleList([create_embedding(self.backbone.out_c, cfg.em_dim) for _ in range(cfg.num_parts)]) #self.backbone.out_c = 2048 cfg.em_dim = 512 cfg.num_parts = 1

    def create_cls_list(self):
        cfg = self.cfg
        self.cls_list = nn.ModuleList([nn.Linear(cfg.em_dim, cfg.num_classes) for _ in range(cfg.num_parts)])# self.cls_list = ModuleList(  (0): Linear(in_features=512, out_features=751, bias=True))
        ori_w = self.cls_list[0].weight.view(-1).detach().numpy().copy() #ori_w是表示self.cls_list[0]的权重参数的最后一列的值
        self.cls_list.apply(init_classifier) #apply(fn)：将fn函数递归地应用到网络模型的每个子模型中，主要用在参数的初始化。使用apply()时，需要先定义一个参数初始化的函数。
        # detach()从当前图中分离出变量，而numpy 表示数组 copy 拷贝
        new_w = self.cls_list[0].weight.view(-1).detach().numpy().copy() #self.cls_list[0] = Linear(in_features=512, out_features=751, bias=True) weight获得参数 view(-1)获得最后一个tensor view()和reshape意思一样，将多行tensor拼成一行，-1表示不知道是具体几行
        import numpy as np
        if np.array_equal(ori_w, new_w): #ori_  w和new_w都是ndarray数组
            from ..utils.log import array_str
            print('!!!!!! Warning: Model Weight Not Changed After Init !!!!!')
            print('Original Weight [:20]:\n\t{}'.format(array_str(ori_w[:20], fmt='{:.6f}')))
            print('New Weight [:20]:\n\t{}'.format(array_str(new_w[:20], fmt='{:.6f}')))

    def get_ft_and_new_params(self, cft=False):
        """cft: Clustering and Fine Tuning"""
        ft_modules, new_modules = self.get_ft_and_new_modules(cft=cft) #ft_modules 是resnet50而new_modules是self.em_list和self.cls_list两个部分
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules])) #ft_params是resnet50的参数
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules])) # new_params 是 new_modules的参数 chain.from iterable接受一个可迭代的参数，返回一个迭代器，m.parameters()返回参数
        return ft_params, new_params #返回获得的参数

    def get_ft_and_new_modules(self, cft=False):
        if cft:
            ft_modules = [self.backbone, self.em_list]
            if hasattr(self, 'ps_head'):
                ft_modules += [self.ps_head]
            new_modules = [self.cls_list] if hasattr(self, 'cls_list') else []
        else:
            ft_modules = [self.backbone] #ft_modules即self.backbone为 list类型，Resnet50
            new_modules = [self.em_list]  #new_modules为self.em_list,类型为Modulelist ，
            if hasattr(self, 'cls_list'):
                new_modules += [self.cls_list]
            if hasattr(self, 'ps_head'):
                new_modules += [self.ps_head]
        return ft_modules, new_modules #ft_modules为self.backbone 即resnet50 new_modules为self.em_list  self.cls_list

    def set_train_mode(self, cft=False, fix_ft_layers=False):
        self.train()
        if fix_ft_layers:
            for m in self.get_ft_and_new_modules(cft=cft)[0]:
                m.eval()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im']) #返回图片的tensor表示

    def reid_forward(self, in_dict):
        # print('=====> in_dict.keys() entering reid_for    ward():', in_dict.keys())
        pool_out_dict = self.pool(in_dict) #对in_dict存储的一个图片进行池化处理，得到6个部分的特征向量
        feat_list = [em(f) for em, f in zip(self.em_list, pool_out_dict['feat_list'])] #em_list是6个处理部位的模型，feat_list是6个图片部分的特征tensor，将tensor赋值给模型
        out_dict = {
            'feat_list': feat_list,#feat_list是list类型的特征，存放着1个图片的6个部分 ，每个部分经过self.em_list模型处理过的tensor
        }
        if hasattr(self, 'cls_list'):
            logits_list = [cls(f) for cls, f in zip(self.cls_list, feat_list)] #cls是模型，f是特征张量 logits_list存放着预测图片特征的概率结果
            out_dict['logits_list'] = logits_list#进来了
        if 'visible' in pool_out_dict:
            out_dict['visible'] = pool_out_dict['visible'] #没有进来
        return out_dict

    def ps_forward(self, in_dict):
        return self.ps_head(in_dict['feat'])

    def forward(self, in_dict, forward_type='reid'):
        in_dict['feat'] = self.backbone_forward(in_dict) #框架的前半部分，网络前向播传，in_dict就是batch数据 in_dict['feat']是in_dict['im']图片信息
        if forward_type == 'reid':#进入到reid
            out_dict = self.reid_forward(in_dict) #框架的后半部分，将前半部分获得的特征进行行人重识别的前向传播
        elif forward_type == 'ps':
            out_dict = {'ps_pred': self.ps_forward(in_dict)}
        elif forward_type == 'ps_reid_parallel': #duke进入到这里了
            out_dict = self.reid_forward(in_dict)
            out_dict['ps_pred'] = self.ps_forward(in_dict)
        elif forward_type == 'ps_reid_serial':
            ps_pred = self.ps_forward(in_dict)
            # Generate pap masks from ps_pred
            in_dict['pap_mask'] = gen_pap_mask_from_ps_pred(ps_pred)
            out_dict = self.reid_forward(in_dict)
            out_dict['ps_pred'] = ps_pred
        else:
            raise ValueError('Error forward_type {}'.format(forward_type))
        return out_dict #out_dict前向传播后的结果
