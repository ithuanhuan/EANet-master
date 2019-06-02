#-*-coding:utf-8-*-
from __future__ import print_function
from collections import OrderedDict
from copy import deepcopy
import torch
from torch.optim.lr_scheduler import MultiStepLR
from package.optim.reid_trainer import ReIDTrainer
from package.model.model import Model
from package.utils.torch_utils import may_data_parallel
from package.utils.torch_utils import recursive_to_device
from package.optim.optimizer import create_optimizer
from package.optim.lr_scheduler import WarmupLR
from package.data.multitask_dataloader import MTDataLoader
from package.loss.triplet_loss import TripletLoss
from package.loss.id_loss import IDLoss
from package.loss.ps_loss import PSLoss


class EANetTrainer(ReIDTrainer):

    def create_train_loader(self, samples=None):
        cfg = self.cfg #{'model': {'backbone': {'name': 'resnet50', 'last_conv_stride': 1, 'pretrained': True, 'pretrained_model_dir': 'imagenet_model'}, 'pool_type': 'GlobalPool', 'max_or_avg': 'max', 'em_dim': 512, 'num_parts': 1, 'use_ps': False, 'ps_head': {'mid_c': 256, 'num_classes': 8}}, 'dataset': {'root': 'dataset', 'im': {'h_w': [256, 128], 'mean': [0.486, 0.459, 0.408], 'std': [0.229, 0.224, 0.225]}, 'use_pap_mask': False, 'pap_mask': {'h_w': [24, 8], 'type': 'PAP_9P'}, 'use_ps_label': False, 'ps_label': {'h_w': [48, 16]}, 'train': {'name': 'market1501', 'split': 'train', 'transform_list': ['hflip', 'resize']}, 'cd_train': {'name': 'duke', 'split': 'train', 'transform_list': ['hflip', 'resize']}, 'test': {'names': ['market1501', 'cuhk03_np_detected_jpg', 'duke'], 'transform_list': ['resize']}}, 'dataloader': {'num_workers': 2, 'train': {'batch_type': 'random', 'batch_size': 32, 'drop_last': True}, 'cd_train': {'batch_type': 'random', 'batch_size': 32, 'drop_last': True}, 'test': {'batch_type': 'seq', 'batch_size': 32, 'drop_last': False}, 'pk': {'k': 4}}, 'eval': {'forward_type': 'reid', 'chunk_size': 1000, 'separate_camera_set': False, 'single_gallery_shot': False, 'first_match_break': True, 'score_prefix': '', 'device': device(type='cpu')}, 'train': {}, 'id_loss': {'name': 'idL', 'weight': 1, 'use': True}, 'tri_loss': {'name': 'triL', 'weight': 0, 'use': False, 'margin': 0.3, 'dist_type': 'euclidean', 'hard_type': 'tri_hard', 'norm_by_num_of_effective_triplets': False}, 'src_ps_loss': {'name': 'psL', 'weight': 0, 'use': False, 'normalize_size': True, 'num_classes': 8}, 'cd_ps_loss': {'name': 'cd_psL', 'weight': 0, 'use': False, 'normalize_size': True, 'num_classes': 8}, 'log': {'use_tensorboard': True, 'exp_dir': '/Users/huan/code/PycharmProjects/EANet-master/exp/eanet/test_paper_models/GlobalPool/market1501', 'ckpt_file': '/Users/huan/code/PycharmProjects/EANet-master/exp/eanet/test_paper_models/GlobalPool/market1501/ckpt.pth', 'score_file': '/Users/huan/code/PycharmProjects/EANet-master/exp/eanet/test_paper_models/GlobalPool/market1501/score_2019-05-27_10-22-52.txt'}, 'optim': {'optimizer': 'sgd', 'sgd': {'momentum': 0.9, 'nesterov': False}, 'weight_decay': 0.0005, 'ft_lr': 0.01, 'new_params_lr': 0.02, 'lr_decay_epochs': [25, 50], 'normal_epochs': 2, 'warmup_epochs': 0, 'warmup': False, 'warmup_init_lr': 0, 'pretrain_new_params_epochs': 0, 'pretrain_new_params': False, 'epochs_per_val': 5, 'steps_per_log': 50, 'trial_run': False, 'phase': 'normal', 'resume': False, 'cft': False, 'cft_iters': 1, 'cft_rho': 0.0008}, 'only_test': False, 'only_infer': False}
        self.train_loader = self.create_dataloader(mode='train', samples=samples)
        if cfg.cd_ps_loss.use:
            self.cd_train_loader = self.create_dataloader(mode='cd_train', samples=samples)
            self.train_loader = MTDataLoader([self.train_loader, self.cd_train_loader], ref_loader_idx=0)

    def create_model(self):
        if hasattr(self, 'train_loader'):#到了这里
            reid_loader = self.train_loader.loaders[0] if self.cfg.cd_ps_loss.use else self.train_loader #到了这里 reid_loader为dukeMTMCreID
            self.cfg.model.num_classes = reid_loader.dataset.num_ids #702
        self.model = Model(deepcopy(self.cfg.model)) #使用resnet50模型，以及增加了em_list和cls_list初始化模型
        self.model = may_data_parallel(self.model) #定义网络结构进去之后什么也没有做
        self.model.to(self.device)

    def set_model_to_train_mode(self):
        cfg = self.cfg.optim
        self.model.set_train_mode(cft=cfg.cft, fix_ft_layers=cfg.phase == 'pretrain') #cft = False

    def create_optimizer(self):
        cfg = self.cfg.optim # self.cfg.optim = easydic类型，是优化器的参数和类型 {'optimizer': 'sgd', 'sgd': {'momentum': 0.9, 'nesterov': False}, 'weight_decay': 0.0005, 'ft_lr': 0.01, 'new_params_lr': 0.02, 'lr_decay_epochs': [25, 50], 'normal_epochs': 2, 'warmup_epochs': 0, 'warmup': False, 'warmup_init_lr': 0, 'pretrain_new_params_epochs': 0, 'pretrain_new_params': False, 'epochs_per_val': 5, 'steps_per_log': 50, 'trial_run': False, 'phase': 'normal', 'resume': False, 'cft': False, 'cft_iters': 1, 'cft_rho': 0.0008}
        ft_params, new_params = self.model.get_ft_and_new_params(cft=cfg.cft) #cfg.cft= false ft_params是resnet50网络模型参数，而new_modules是self.em_list和self.cls_list两个部分的参数
        if cfg.phase == 'pretrain':
            assert len(new_params) > 0, "No new params to pretrain!"
            param_groups = [{'params': new_params, 'lr': cfg.new_params_lr}]
        else: #cfg.phase = 'normal'
            param_groups = [{'params': ft_params, 'lr': cfg.ft_lr}] #ft_params是模型结构的参数 cfg.ft_lr = 0.01
            # Some model may not have new params
            if len(new_params) > 0:#到了这里
                param_groups += [{'params': new_params, 'lr': cfg.new_params_lr}] #cfg.new_params_lr = 0.02
        self.optimizer = create_optimizer(param_groups, cfg)  # self.optimizer = {SGD} SGD(Parameter Group0:  )一系列参数
        recursive_to_device(self.optimizer.state_dict(), self.device) #optimizer是训练的工具

    def create_lr_scheduler(self):
        cfg = self.cfg.optim
        if cfg.phase == 'normal': # 值为normal
            cfg.lr_decay_steps = [len(self.train_loader) * ep for ep in cfg.lr_decay_epochs] #ep即cfe.lr_decay_step的值为[25, 50] cfg.lr_decay_steps的值为[10100, 20200]
            cfg.epochs = cfg.normal_epochs # 2
            self.lr_scheduler = MultiStepLR(self.optimizer, cfg.lr_decay_steps) #学习率调整
        elif cfg.phase == 'warmup':
            cfg.warmup_steps = cfg.warmup_epochs * len(self.train_loader)
            cfg.epochs = cfg.warmup_epochs
            self.lr_scheduler = WarmupLR(self.optimizer, cfg.warmup_steps)
        elif cfg.phase == 'pretrain':
            cfg.pretrain_new_params_steps = cfg.pretrain_new_params_epochs * len(self.train_loader)
            cfg.epochs = cfg.pretrain_new_params_epochs
            self.lr_scheduler = None
        else:
            raise ValueError('Invalid phase {}'.format(cfg.phase))

    def create_loss_funcs(self):
        cfg = self.cfg
        self.loss_funcs = OrderedDict() #使用OrderedDict会根据放入元素的先后顺序进行排序
        if cfg.id_loss.use:#进入到这里
            self.loss_funcs[cfg.id_loss.name] = IDLoss(cfg.id_loss, self.tb_writer) # self.loss_funcs['idL'] =
        if cfg.tri_loss.use:
            self.loss_funcs[cfg.tri_loss.name] = TripletLoss(cfg.tri_loss, self.tb_writer)
        if cfg.src_ps_loss.use: #进入到这里
            self.loss_funcs[cfg.src_ps_loss.name] = PSLoss(cfg.src_ps_loss, self.tb_writer)
        if cfg.cd_ps_loss.use: #进入到这里
            self.loss_funcs[cfg.cd_ps_loss.name] = PSLoss(cfg.cd_ps_loss, self.tb_writer)

    # NOTE: To save GPU memory, our multi-domain training requires
    # [1st batch: source-domain forward and backward]-
    # [2nd batch: cross-domain forward and backward]-
    # [update model]
    # So the following three-step framework is not strictly followed.
    #     pred = self.train_forward(batch)
    #     loss = self.criterion(batch, pred)
    #     loss.backward()
    def train_forward(self, batch):
        cfg = self.cfg
        batch = recursive_to_device(batch, self.device) #batch包括DukeMTMC一个batch数据的字典 和 Market1501一个batch数据的字典 self.device = cpu，如果cuda可用则使用gpu bach为dict字典类型，四个键分别 im_path值为图片路径 label标签 cam摄像头 im图片
        if cfg.cd_ps_loss.use:
            reid_batch, cd_ps_batch = batch #reid_batch是源域DukeMTMC的一个batch数据的字典 和 cd_ps_batch是目标域Market1501一个batch数据的字典
        else:
            reid_batch = batch #进入到这里了
        # Source Loss 源域的前向传播和反向传播
        loss = 0
        pred = self.model.forward(reid_batch, forward_type='ps_reid_parallel' if cfg.src_ps_loss.use else 'reid')#pred = {dict}[feat_list = {list}一个batch中所有图片预测的概率值, logits_list={list}一个batch中所有图片的特征]
        for loss_cfg in [cfg.id_loss, cfg.tri_loss, cfg.src_ps_loss]:#duke使用了两个损失id_loss = true src和ps_loss = true
            if loss_cfg.use:
                loss += self.loss_funcs[loss_cfg.name](reid_batch, pred, step=self.trainer.current_step)['loss'] # 32张图片loss_funcs(真实值，预测值)
        if isinstance(loss, torch.Tensor):
            loss.backward() #误差反向传播，计算参数更新值 反向传播具体如何更新参数的
        # Cross-Domain Loss 跨域的前向传播和反向传播
        if cfg.cd_ps_loss.use:#进入到这里
            pred = self.model.forward(cd_ps_batch, forward_type='ps')
            loss = self.loss_funcs[cfg.cd_ps_loss.name](cd_ps_batch, pred, step=self.trainer.current_step)['loss']
            if isinstance(loss, torch.Tensor):
                loss.backward()

    def criterion(self, batch, pred):
        return 0

    def train_phases(self):
        cfg = self.cfg.optim
        if cfg.warmup:
            cfg.phase = 'warmup'
            cfg.dont_test = True
            self.init_trainer()
            self.train()
            cfg.phase = 'normal'
            cfg.dont_test = False
            self.init_trainer()
            self.load_items(model=True, optimizer=True)
        self.train()


if __name__ == '__main__':
    from package.utils import init_path
    #todo 先注释其他的
    trainer = EANetTrainer()
    if trainer.cfg.only_test:
        trainer.test()
    else:
        trainer.train_phases() #训练
