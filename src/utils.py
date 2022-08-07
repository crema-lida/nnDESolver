import numpy as np
import torch


def gen_inputs(domain, steps) -> (np.ndarray, dict[str, np.ndarray]):
    arrs = []  # 1d arrays of coordinates
    shape = []  # length of each dimension
    steps = steps if type(steps) is tuple else (steps,) * len(domain)
    for limits, step in zip(domain.values(), steps):
        arr = np.arange(limits[0], limits[1] + step, step, dtype='float32')
        arrs.append(arr)
        shape.append(arr.size)
    assert np.prod(shape) < 30000000, 'Sample size is too large. Please try with bigger step.'
    mesh = np.meshgrid(*arrs, indexing='ij')
    inputs = np.array(list(zip(*map(lambda v: v.flat, mesh))), dtype='float32')
    return inputs, dict(zip(domain.keys(), arrs))


def downsample(inputs, coords) -> (np.ndarray, dict[str, np.ndarray]):
    """sample size of each dimension will be reduced to roughly 100 (1-dim) or 50 (multi-dim)"""
    ds_coords = {name: np.full_like(arr, np.nan) for name, arr in coords.items()}
    count = 100 if len(coords) == 1 else 50
    for arr, ds_arr in zip(coords.values(), ds_coords.values()):
        stride = max(1, round(arr.size / count))
        ds_arr[::stride] = arr[::stride]
    ds_mesh = np.meshgrid(*ds_coords.values(), indexing='ij', copy=False)
    ds_input = np.array(list(zip(*map(lambda v: v.flat, ds_mesh))), dtype='float32')
    choice = ~np.fromiter(map(lambda item: item.any(), np.isnan(ds_input)), dtype=bool)
    indices = np.full_like(inputs[:, 0], -1, dtype=int)
    indices[choice] = np.arange(choice[choice].size)  # index exists where choice is True, else index is `-1`(unwanted)
    return indices, {name: arr[~np.isnan(arr)] for name, arr in ds_coords.items()}


def parse_inputs(args):
    if not args:
        order = ''
        position = ()
    elif type(args[0]) is str:
        order = args[0]
        position = args[1:]
    else:
        order = ''
        position = args
    return order, position


def get_bdy_pos(position, device):
    has_tensor = False
    len_inputs = 0
    for coord in position:
        if type(coord) is torch.Tensor:
            has_tensor = True
            len_inputs = coord.size(0)
    if has_tensor:
        inputs = []
        for coord in position:
            if type(coord) is torch.Tensor:
                inputs.append(coord)
            else:
                inputs.append(torch.tensor([[coord]], device=device).expand(len_inputs, -1))
    else:
        inputs = [torch.tensor([[coord]]) for coord in position]
    return torch.cat(inputs, dim=1).to(device=device, dtype=torch.float).requires_grad_()
