import numpy as np
import re


def generate_inputs(domain, step) -> (np.ndarray, list, list):
    assert (T := type(domain)) is dict, f'Expected domain type: `tuple` or `dict`. However, type `{T}` is given.'
    var_names = list(domain.keys())
    coords = []  # 1d arrays of coordinates
    shape = []  # length of each dimension
    for limits in domain.values():
        arr = np.arange(limits[0], limits[1] + step, step, dtype='float32')
        coords.append(arr)
        shape.append(arr.size)
    if np.prod(shape) > 30000000:
        raise Exception('Sample size is too large. Please try with larger step.')
    mesh = np.meshgrid(*coords, indexing='ij')
    inputs = np.array(list(zip(*map(lambda v: v.flat, mesh))), dtype='float32')
    return inputs, var_names, coords


def downsample(inputs, coords) -> (np.ndarray, list):
    """sample size of each dimension will be down-sampled to roughly 100 (1-dim) or 50 (multi-dim)"""
    ds_coords = [np.full_like(arr, np.nan) for arr in coords]
    count = 100 if len(coords) == 1 else 50
    for arr, ds_arr in zip(coords, ds_coords):
        stride = max(1, round(arr.size / count))
        ds_arr[::stride] = arr[::stride]
    ds_mesh = np.meshgrid(*ds_coords, indexing='ij', copy=False)
    ds_input = np.array(list(zip(*map(lambda v: v.flat, ds_mesh))), dtype='float32')
    choice = ~np.fromiter(map(lambda item: item.any(), np.isnan(ds_input)), dtype=bool)
    indices = np.full_like(inputs[:, 0], -1, dtype=int)
    indices[choice] = np.arange(choice[choice].size)  # index exists where choice is True, else index is `-1`(unwanted)
    return indices, [arr[~np.isnan(arr)] for arr in ds_coords]


def translate(functions, var_names, *expressions) -> None:
    mathfunc = ['sin', 'cos', 'tan', 'exp', 'log', 'log2', 'log10', 'sqrt', 'arctan']
    mathconst = ['pi', 'e']

    def replacement(match):
        if f := match.group('func'):
            if f not in functions.keys():
                functions[f] = None
            return '{}("{}")'.format(f, match.group('order'))
        if (m := match.group('var')) in var_names:
            index = var_names.index(m)
            return 'Func.inputs[:, {}:{}]'.format(index, index + 1)
        if m in mathfunc + mathconst:
            return f'torch.{m}'
        return match.group()

    pattern = re.compile(r'\b(?P<func>\w)\((?P<order>.*?)\)|(?<!\.)\b(?P<var>\w+)\b')
    for each in expressions:
        for i, expr in enumerate(each):
            each[i] = pattern.sub(replacement, expr)
