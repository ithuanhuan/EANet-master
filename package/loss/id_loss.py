from __future__ import print_function
import torch
from .loss import Loss
from ..utils.meter import RecentAverageMeter as Meter


class IDLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(IDLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')  #交叉墒损失

    def __call__(self, batch, pred, step=0, **kwargs):
        cfg = self.cfg

        # Calculation
        if 'visible' in pred:
            loss_list = [(self.criterion(logits, batch['label']) * pred['visible'][:, i]).sum()
                         / (pred['visible'][:, i].sum() + 1e-12)
                         for i, logits in enumerate(pred['logits_list'])]
        else:#进入到这里了
            loss_list = [self.criterion(logits, batch['label']).mean() for logits in pred['logits_list']] #criterion(output, target)计算交叉墒损失,mean()返回loss和的平均值
        # New version of pytorch allow stacking 0-dim tensors, but not concatenating.
        loss = torch.stack(loss_list).sum() #stack()对张量进行连接，sum()求和

        # Meter
        if cfg.name not in self.meter_dict:
            self.meter_dict[cfg.name] = Meter(name=cfg.name)
        self.meter_dict[cfg.name].update(loss.item())#loss.item() = float类型的6.62163
        if len(loss_list) > 1:
            part_fmt = '#{}'
            for i in range(len(loss_list)):
                if part_fmt.format(i + 1) not in self.meter_dict:
                    self.meter_dict[part_fmt.format(i + 1)] = Meter(name=part_fmt.format(i + 1))
                self.meter_dict[part_fmt.format(i + 1)].update(loss_list[i].item())

        # Tensorboard
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(cfg.name, {cfg.name: self.meter_dict[cfg.name].avg}, step)
            if len(loss_list) > 1:
                self.tb_writer.add_scalars('Part ID Losses', {part_fmt.format(i + 1): self.meter_dict[part_fmt.format(i + 1)].avg for i in range(len(loss_list))}, step)

        # Scale by loss weight
        loss *= cfg.weight

        return {'loss': loss}
