"""
    This is a fork from: https://github.com/graykode/modelsummary
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from pytorch_model_summary.hierarchical_summary import hierarchical_summary


def summary(model, *inputs, batch_size=-1, show_input=False, show_hierarchical=False, 
            print_summary=False):

    def register_id_parent_id(module):
        _map_modulelist_ids = dict()

        def _register(module):
            for c in module.children():
                map_id_parent_id[id(c)] = id(module)
                register_id_parent_id(c)
                if isinstance(c, nn.ModuleList):
                    _map_modulelist_ids[id(c)] = id(module)

        _register(module)

        if len(_map_modulelist_ids):
            for m in map_id_parent_id.keys():
                if map_id_parent_id.get(m) in _map_modulelist_ids:
                    map_id_parent_id[m] = _map_modulelist_ids.get(map_id_parent_id.get(m))

    def register_hook(module):
        
        def shapes(x):
            _lst = list()

            def _shapes(_):
                if isinstance(_, torch.Tensor):
                    _lst.append(list(_.size()))
                elif isinstance(_, (tuple, list)):
                    for _x in _:
                        _shapes(_x)
                else:
                    # TODO: decide what to do when there is an input which is not a tensor
                    raise Exception('Object not supported')

            _shapes(x)

            return _lst

        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            summary[m_key]["input_shape"] = shapes(input) if len(input) != 0 else input

            if show_input is False and output is not None:
                summary[m_key]["output_shape"] = shapes(output)

            params = 0
            params_trainable = 0
            trainable = False
            for m in module.parameters():
                _params = torch.prod(torch.LongTensor(list(m.size())))
                params += _params
                params_trainable += _params if m.requires_grad else 0
                # if any parameter is trainable, then module is trainable
                trainable = trainable or m.requires_grad

            summary[m_key]["nb_params"] = params
            summary[m_key]["nb_params_trainable"] = params_trainable
            summary[m_key]["trainable"] = trainable

        if not isinstance(module, nn.Sequential) and id(module) != id(model) and map_id_parent_id[id(module)] == id(model):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    map_id_parent_id = dict()
    hooks = []

    # register id of parent modules
    register_id_parent_id(model)

    # register hook
    model.apply(register_hook)
    out = model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    # params to format output
    _key_shape = 'input_shape' if show_input else 'output_shape'
    _len_str_layer = max([len(layer) for layer in summary.keys()] + [15]) + 3
    _len_str_shapes = max([len(', '.join([str(_) for _ in summary[layer][_key_shape]])) for layer in summary] + [15]) + 3
    _len_line = 34 + _len_str_layer + _len_str_shapes
    fmt = "{:>%d}  {:>%d} {:>15} {:>15}" % (_len_str_layer, _len_str_shapes)

    """ starting to build output text """

    # Table header
    lines = list()
    lines.append('-' * _len_line)
    lines.append(fmt.format("Layer (type)", "Input Shape" if show_input else "Output Shape", "Param #", "Tr. Param #"))
    lines.append('=' * _len_line)

    total_params = 0
    trainable_params = 0
    for layer in summary:
        # Table content (for each layer)
        line_new = fmt.format(
                layer,
                ', '.join([str(_) for _ in summary[layer][_key_shape]]),
                "{0:,}".format(summary[layer]["nb_params"]),
                "{0:,}".format(summary[layer]["nb_params_trainable"]),
            )
        lines.append(line_new)

        total_params += summary[layer]["nb_params"]
        trainable_params += summary[layer]["nb_params_trainable"]

    # Table footer
    lines.append('=' * _len_line)
    lines.append("Total params: {0:,}".format(total_params))
    lines.append("Trainable params: {0:,}".format(trainable_params))
    lines.append("Non-trainable params: {0:,}".format(total_params - trainable_params))
    if batch_size != -1:
        lines.append("Batch size: {0:,}".format(batch_size))
    lines.append('-' * _len_line)

    if show_hierarchical:
        h_summary, _ = hierarchical_summary(model, print_summary=False)

        # Building hierarchical output
        _pad = int(max(max(len(_) for _ in h_summary.split('\n')) - 20, 0) / 2)
        lines.append('\n\n' + '=' * _pad + ' Hierarchical Summary ' + '=' * _pad + '\n')
        lines.append(h_summary)
        lines.append('\n\n' + '=' * (_pad * 2 + 22) + '\n')

    str_summary = '\n'.join(lines)
    if print_summary:
        print(str_summary)

    return str_summary
