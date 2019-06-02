#-*-coding:utf-8-*-
from __future__ import print_function
import torch


class Trainer(object):
    def __init__(self, train_loader, train_forward, criterion, optimizer, lr_scheduler,
                 steps_per_log=1, print_step_log=None, eps_per_log=1, print_ep_log=None):
        self.train_loader = train_loader #dukeMTMC和market1501
        self.train_forward = train_forward
        self.criterion = criterion
        self.optimizer = optimizer #SGD优化器
        self.lr_scheduler = lr_scheduler #
        self.steps_per_log = steps_per_log #50
        self.print_step_log = print_step_log
        self.eps_per_log = eps_per_log #1
        self.print_ep_log = print_ep_log
        self.current_step = 0  # will NOT be reset between epochs
        self.current_ep = 0

    def train_one_step(self, batch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step() #调整学习率
        self.optimizer.zero_grad() #梯度grad初始化为零
        pred = self.train_forward(batch) #pred = None
        loss = self.criterion(batch, pred) #loss = 0
        # some/all backward may be done in self.criterion/self.train_forward
        if isinstance(loss, torch.Tensor):
            loss.backward() #反向传播求梯度，计算参数更新值
        self.optimizer.step() #更新所有参数
        if ((self.current_step + 1) % self.steps_per_log == 0) and (self.print_step_log is not None):
            self.print_step_log()
        self.current_step += 1

    def train_one_epoch(self, trial_run_steps=None):
        for i, batch in enumerate(self.train_loader): #batch是训练一次的一组数据
            self.train_one_step(batch)
            # 'Trial Run' to quickly make sure there is no error
            if (trial_run_steps is not None) and (i + 1 >= trial_run_steps):
                break
        if ((self.current_ep + 1) % self.eps_per_log == 0) and (self.print_ep_log is not None):
            self.print_ep_log()
        self.current_ep += 1

    def reset_counters(self):
        self.current_step = 0
        self.current_ep = 0
