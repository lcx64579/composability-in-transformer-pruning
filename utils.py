import torch
import torch.nn as nn

# 替换模型中指定的一层
def set_module(model: nn.Module, submodule_key: str, module: nn.Module) -> None:
    r"""
    替换模型中指定的一层。

    Args:
        `model`: 模型
        `submodule_key`: 一个字符串，指定哪个层要替换。例如AlexNet中的`feature.3`，
        或者Transformer中的`transformer.encoder.layers.0.linear1`
        `module`: 替换上去的层

    Output:
        无
    """
    # 核心函数，参考了torch.quantization.fuse_modules()的实现
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)