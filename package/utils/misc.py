import numpy as np
import itertools


def concat_dict_list(dict_list): #dict_list是长度为19732的list，list的每一个值是字典 'im_path'：'图片路径', 'feat':'图片特征'，
    ret_dict = {}
    keys = dict_list[0].keys() #keys = dict_keys(['im_path', 'feat'])
    for k in keys: #k 的值分别为 'im_path', 'feat'
        if isinstance(dict_list[0][k], list):#进入到这里
            ret_dict[k] = list(itertools.chain.from_iterable([dict_[k] for dict_ in dict_list])) #ret_dict['im_path'] = {list} 19732张图片的路径 00000 = '路径'，00001 = '路径'
        elif isinstance(dict_list[0][k], np.ndarray):
            ret_dict[k] = np.concatenate([dict_[k] for dict_ in dict_list]) #ret_dict['feat'] = {ndarray} 19732张图片的特征
        else:
            raise NotImplementedError
    return ret_dict


def import_file(path):
    import sys, importlib
    import os.path as osp
    path_to_insert = osp.dirname(osp.abspath(osp.expanduser(path)))
    sys.path.insert(0, path_to_insert)
    imported = importlib.import_module(osp.splitext(osp.basename(path))[0])
    # NOTE: sys.path may be modified inside the imported file. If path_to_insert
    # is not added to sys.path at any index inside the imported file, this remove()
    # can exactly cancel the previous insert(). Otherwise, the element order inside
    # sys.path may not be desired by the imported file.
    sys.path.remove(path_to_insert)
    return imported
