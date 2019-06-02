import re
import pickle
#src = {'name': 'market1501', 'split': 'train', 'transform_list': ['hflip', 'resize']}
#dest = {'root': 'dataset', 'im': {'h_w': [256, 128], 'mean': [0.486, 0.459, 0.408], 'std': [0.229, 0.224, 0.225]}, 'use_pap_mask': False, 'pap_mask': {'h_w': [24, 8], 'type': 'PAP_9P'}, 'use_ps_label': False, 'ps_label': {'h_w': [48, 16]}, 'train': {'name': 'market1501', 'split': 'train', 'transform_list': ['hflip', 'resize']}, 'cd_train': {'name': 'duke', 'split': 'train', 'transform_list': ['hflip', 'resize']}, 'test': {'names': ['market1501', 'cuhk03_np_detected_jpg', 'duke'], 'transform_list': ['resize']}}
def transfer_items(src, dest):
    for k, v in src.items(): # k = name v = market1501 ; k = split v = train ; k = transform_list v = ['hflip', 'resize']
        dest[k] = src[k] # desk[name] = 'market1501' desk[split] = train dest[transform_list] = ['hflip','resize']


def overwrite_cfg_file(cfg_file, ow_str='None', ow_file='None', new_cfg_file='None'):
    """Overwrite some items of a EasyDict defined config file.
    Args:
        cfg_file: The original config file
        ow_str: Mutually exclusive to ow_file. Specify the new items (separated by ';') to overwrite.
            E.g. "cfg.model = 'ResNet-50'; cfg.im_mean = (0.5, 0.5, 0.5)".
        ow_file: A text file, each line being a new item.
        new_cfg_file: Where to write the updated config. If 'None', overwrite the original file.
    """

    with open(cfg_file, 'r') as f:
        lines = f.readlines()
    if ow_str != 'None':
        cfgs = ow_str.split(';')
        cfgs = [cfg.strip() for cfg in cfgs if cfg.strip()]
    else:
        with open(ow_file, 'r') as f:
            cfgs = f.readlines()
        # Skip empty or comment lines
        cfgs = [cfg.strip() for cfg in cfgs if cfg.strip() and not cfg.strip().startswith('#')]
    for cfg in cfgs:
        key, value = cfg.split('=')
        key = key.strip()
        value = value.strip()
        pattern = r'{}\s*=\s*(.*?)(\s*)(#.*)?(\n|$)'.format(key.replace('.', '\.'))
        def func(x):
            # print(r'=====> {} groups, x.groups(): {}'.format(len(x.groups()), x.groups()))
            # x.group(index), index starts from 1
            # x.group(index) may be `None`
            # x.group(4) is either '\n' or ''
            return '{} = {}'.format(key, value) + (x.group(2) or '') + (x.group(3) or '') + x.group(4)
        new_lines = []
        for line in lines:
            # Skip empty or comment lines
            if not line.strip() or line.strip().startswith('#'):
                new_lines.append(line)
                continue
            line = re.sub(pattern, func, line)
            new_lines.append(line)
        lines = new_lines
    if new_cfg_file == 'None':
        new_cfg_file = cfg_file
    with open(new_cfg_file, 'w') as f:
        # f.writelines(lines)  # Same effect
        f.write(''.join(lines))