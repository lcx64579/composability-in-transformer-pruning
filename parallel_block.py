import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ParallelFinetuningBlock(nn.Module):
    """构造一个原块与一个剪枝块并联的结构，其中原块的输出用作该模块的输出。

    Args:
        `original_module`: 原块
        `pruned_module`: 剪枝块
        `module_type`: 块类型。为一字符串。只能为`"Linear"`或`"MultiheadAttention"`。
        `head_mask`: 如果`module_type`为`"MultiheadAttention"`，则应为`pruned_module`的
        剪枝mask，否则传入`None`

    Attributes:
        `original_module`: 原块
        `pruned_module`: 剪枝块
        `out_original`: 原块的输出，用于Loss计算
        `out_pruned`: 剪枝块的输出，用于Loss计算
        `module_type`: 块类型。为一字符串。只能为`"Linear"`或`"MultiheadAttention"`。
        `head_mask`: 如果`module_type`为`"MultiheadAttention"`，则应为`pruned_module`的
        剪枝mask，否则该属性不存在

    Forward:
        Input: `*input`, `**kwargs`
        Output: `original_module(*input, **kwargs)[0]`
    """

    def __init__(self, original_module, pruned_module, module_type, head_mask):
        super(ParallelFinetuningBlock, self).__init__()
        self.original_module = original_module
        self.pruned_module = pruned_module
        self.out_original = None
        self.out_pruned = None
        assert module_type == "Linear" or module_type == "MultiheadAttention", "Module must be Linear or MultiheadAttention"
        self.module_type = module_type
        if module_type == "MultiheadAttention":
            self.head_mask = head_mask       # 2022-10-09：新加的，在初始化的时候传过来。只是方便查询而已，在forward用不到。

    def forward(self, *input, **kwargs):
        z_orig = self.original_module(*input, **kwargs)
        z_pruned = self.pruned_module(*input, **kwargs)
        self.out_original = z_orig[0]       # 其输出为 (attn_output, attn_output_weights)，第二项不需要
        self.out_pruned = z_pruned[0]
        return z_orig[0]


def construct_parallel_block(model: nn.Module, name_pruned: str, module_pruned: nn.Module, module_type: str, head_mask: torch.Tensor = None) -> nn.Module:
    r"""
    构造一个「并联」结构，一边是原块，另一边是剪枝了的块。

    注意: 不要在微调前深拷贝被剪块，不然之后还要想办法拿到微调好的被剪块。

    Args:
        `model`: 原模型，用于找原块
        `name_pruned`: 被剪块的名字，用于在原模型中定位原块
        `module_pruned`: 被剪块，用于并联

    Output:
        「并联」结构，并且结构中有个存两边各自输出的属性。
    """
    # 先从原模型中找原块
    tokens = name_pruned.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    # 现在cur_mod是原块
    # 然后用 ParallelFinetuningBlock 类制作并联块
    new_block = ParallelFinetuningBlock(cur_mod, module_pruned, module_type, head_mask)     # 没有head_mask时，默认为None
    return new_block
