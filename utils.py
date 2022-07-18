import torch
import re


def generate_inputs(domain, step):
    if type(domain) != dict:
        raise Exception(f'Expected domain type: `tuple` or `dict`. However, type `{type(domain)}` is given.')
    var_names = list(domain.keys())
    var_list = []
    shape = []
    for limits in domain.values():
        values = torch.arange(limits[0], limits[1] + step, step)
        var_list.append(values)
        shape.append(values.size(0))
    inputs = torch.empty(*shape, len(shape))
    dimension = range(len(shape))
    index = [0 for i in shape]
    max_index = [i - 1 for i in shape[:-1]] + shape[-1:]
    while index < max_index:
        for i in dimension:
            inputs[tuple(index)][i] = var_list[i][index[i]]
        index[-1] += 1
        for i in dimension[:0:-1]:
            if index[i] == shape[i]:
                index[i] = 0
                index[i - 1] += 1
    return inputs.view(-1, len(shape)).to(dtype=torch.float), var_names, tuple(shape)


def translate(functions, var_names, *expressions):
    mathfunc = ['sin', 'cos', 'tan', 'exp', 'log', 'log2', 'log10', 'sqrt', 'arctan']
    mathconst = ['pi', 'e']

    def replacement(match):
        if f := match.group('func'):
            if f not in functions.keys():
                functions[f] = None
            return '{}("{}")'.format(f, match.group('order'))
        if (m := match.group('var')) in var_names:
            index = var_names.index(m)
            return 'self.inputs[:, {}:{}]'.format(index, index + 1)
        if m in mathfunc + mathconst:
            return f'torch.{m}'
        return match.group()

    pattern = re.compile(r'\b(?P<func>\w)\((?P<order>.*?)\)|(?<!\.)\b(?P<var>\w+)\b')
    for each in expressions:
        for i, expr in enumerate(each):
            each[i] = pattern.sub(replacement, expr)
