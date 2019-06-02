import torch.nn as nn


class GlobalPool(object):
    def __init__(self, cfg):
        self.pool = nn.AdaptiveAvgPool2d(1) if cfg.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d(1) #nn.AdaptiveAvgPool2d(1), 会返回1*1的池化结果

    def __call__(self, in_dict):
        feat = self.pool(in_dict['feat'])
        feat = feat.view(feat.size(0), -1) #将多维度的tensor展平成为一维 feat.size(0)指batch_size的值 view和reshape功能一样，-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        out_dict = {'feat_list': [feat]}
        return out_dict #{dict}{'feat_list':[tensor(1.5)]} 图片信息
