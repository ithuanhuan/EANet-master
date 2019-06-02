from __future__ import print_function
from tqdm import tqdm
import torch
from .torch_distance import normalize
from ..utils.misc import concat_dict_list
from ..utils.torch_utils import recursive_to_device


def extract_batch_feat(model, in_dict, cfg):
    model.eval() #让模型变成测试模型，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
    with torch.no_grad():#True不反向传播，不求导
        in_dict = recursive_to_device(in_dict, cfg.device)
        out_dict = model(in_dict, forward_type=cfg.forward_type)
        out_dict['feat_list'] = [normalize(f) for f in out_dict['feat_list']] #f为某一个图片的特征张量，normalize为正则化， out_dict['feat_list']一个图片6个部分的特征
        feat = torch.cat(out_dict['feat_list'], 1)#按照维度1进行拼接，即横着拼 feat:tensor([[0.0018, 0.0401, 0.0427,  ..., 0.0000, 0.0002, 0.0000]])
        feat = feat.cpu().numpy()#先转换成为cpu的tensor，再转到numpy格式
        ret_dict = {
            'im_path': in_dict['im_path'],
            'feat': feat,
        }#字典存储着图片的路径，图片的6个部分特征
        if 'label' in in_dict:
            ret_dict['label'] = in_dict['label'].cpu().numpy()
        if 'cam' in in_dict:
            ret_dict['cam'] = in_dict['cam'].cpu().numpy()
        if 'visible' in out_dict:
            ret_dict['visible'] = out_dict['visible'].cpu().numpy()
    return ret_dict # 字典类型，一个图片路径和6个部分连接起来的图片特征


def extract_dataloader_feat(model, loader, cfg):
    dict_list = []
    for batch in tqdm(loader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
        feat_dict = extract_batch_feat(model, batch, cfg)
        dict_list.append(feat_dict)
    ret_dict = concat_dict_list(dict_list)
    return ret_dict
