import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

def remove_prefix(load_state_dict, name):
  new_load_state_dict = dict()
  for key in load_state_dict.keys():
    if key.startswith(name):
      dst_key = key.replace(name, '')
    else:
      dst_key = key
    new_load_state_dict[dst_key] = load_state_dict[key]
  load_state_dict = new_load_state_dict
  return load_state_dict


# load_pretrained ------------------------------------
def load_pretrained_state_dict(net, load_state_dict, strict=False, can_print=True):
  if 'epoch' in load_state_dict and can_print:
    epoch = load_state_dict['epoch']
    print(f'load epoch:{epoch:.2f}')
  if 'state_dict' in load_state_dict:
    load_state_dict = load_state_dict['state_dict']
  elif 'model_state_dict' in load_state_dict:
    load_state_dict = load_state_dict['model_state_dict']
  elif 'model' in load_state_dict:
    load_state_dict = load_state_dict['model']
  if isinstance(net, (DataParallel, DistributedDataParallel)):
    state_dict = net.module.state_dict()
  else:
    state_dict = net.state_dict()

  load_state_dict = remove_prefix(load_state_dict, 'module.')
  load_state_dict = remove_prefix(load_state_dict, 'base_model.')

  for key in list(load_state_dict.keys()):
    if key not in state_dict:
      if strict:
        raise Exception(f'not in {key}')
      if can_print:
        print('not in', key)
      continue
    if load_state_dict[key].size() != state_dict[key].size():
      if strict or (len(load_state_dict[key].size()) != len(state_dict[key].size())):
        raise Exception(f'size not the same {key}: {load_state_dict[key].size()} -> {state_dict[key].size()}')
      if can_print:
        print(f'{key} {load_state_dict[key].size()} -> {state_dict[key].size()}')
      state_slice = [slice(s) for s in np.minimum(np.array(load_state_dict[key].size()), np.array(state_dict[key].size()))]
      state_dict[key][state_slice] = load_state_dict[key][state_slice]
      continue
    state_dict[key] = load_state_dict[key]

  if isinstance(net, (DataParallel, DistributedDataParallel)):
    net.module.load_state_dict(state_dict)
  else:
    net.load_state_dict(state_dict)
  return net

def load_pretrained(net, pretrained_file, strict=False, can_print=False):
  if can_print:
    print(f'load pretrained file: {pretrained_file}')
  load_state_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))
  net = load_pretrained_state_dict(net, load_state_dict, strict=strict, can_print=can_print)
  return net