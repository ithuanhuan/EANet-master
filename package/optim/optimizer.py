import torch.optim as optim


def create_optimizer(param_groups, cfg):
    if cfg.optimizer == 'sgd': # 走到这里了 是等于'sgd'
        optim_class = optim.SGD
    elif cfg.optimizer == 'adam':
        optim_class = optim.Adam
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(cfg.optimizer))
    optim_kwargs = dict(weight_decay=cfg.weight_decay) # optim_kwargs = {'weight_decay': 0.0005}
    if cfg.optimizer == 'sgd':
        optim_kwargs['momentum'] = cfg.sgd.momentum # 0.9
        optim_kwargs['nesterov'] = cfg.sgd.nesterov # false

    return optim_class(param_groups, **optim_kwargs)
