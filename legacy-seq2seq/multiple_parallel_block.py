import torch
import torch.nn as nn

class MultipleParallelFinetuningBlock(nn.Module):
    """构造一个原块与多个剪枝块并联的结构，其中原块的输出用作该模块的输出。

    Args:
        `original_module`: 原块
        `pruned_module_list`: 剪枝块的列表
        `module_type`: 块类型。为一字符串。只能为`"Linear"`或`"MultiheadAttention"`
        `head_mask_list`: 如果`module_type`为`"MultiheadAttention"`，则应为`pruned_module_list`中
        每一块的剪枝mask的列表，否则传入`None`或空列表`[]`

    Attributes:
        `original_module`: 原块
        `pruned_module_list`: 剪枝块的列表
        `out_original`: 原块的输出，用于Loss计算
        `out_pruned_list`: 剪枝块的输出的列表，用于Loss计算
        `module_type`: 块类型。为一字符串。只能为`"Linear"`或`"MultiheadAttention"`
        `head_mask_list`: 如果`module_type`为`"MultiheadAttention"`，则应为`pruned_module_list`中
        每一块的剪枝mask的列表，否则该属性不存在

    Forward:
        Input: `*input`, `**kwargs`
        Output: `original_module(*input, **kwargs)[0]`
    """

    def __init__(self, original_module, pruned_module_list, module_type, head_mask_list):
        super(MultipleParallelFinetuningBlock, self).__init__()
        self.original_module = original_module
        assert isinstance(pruned_module_list, list)
        self.pruned_module_list = pruned_module_list
        self.out_original = None
        self.out_pruned_list = []
        assert module_type == "Linear" or module_type == "MultiheadAttention", "Module must be Linear or MultiheadAttention"
        self.module_type = module_type
        if module_type == "MultiheadAttention":
            assert isinstance(head_mask_list, list)
            assert len(head_mask_list) == len(pruned_module_list)
            assert isinstance(head_mask_list[0], torch.Tensor)
            self.head_mask_list = head_mask_list       # 在初始化的时候传过来。只是方便查询而已，在forward用不到。

    def forward(self, *input, **kwargs):
        z_orig = self.original_module(*input, **kwargs)
        self.out_original = z_orig[0]       # 其输出为 (attn_output, attn_output_weights)，第二项不需要
        self.out_pruned_list = []           # 必须清零！否则第二次forward的结果会跟在第一次后面
        for pruned_module in self.pruned_module_list:
            z_pruned = pruned_module(*input, **kwargs)
            self.out_pruned_list.append(z_pruned[0])
        return z_orig[0]


def construct_parallel_block(model: nn.Module, name_pruned: str, module_pruned_list: list, module_type: str, head_mask_list: list = None) -> nn.Module:
    r"""
    构造一个「并联」结构，一边是原块，另一边是剪枝了的块的列表。

    注意: 不要在微调前深拷贝被剪块，不然之后还要想办法拿到微调好的被剪块。

    Args:
        `model`: 原模型，用于找原块
        `name_pruned`: 被剪块的名字，用于在原模型中定位原块
        `module_pruned_list`: 被剪块的列表，用于并联
        `module_type`: 块类型。为一字符串。只能为`"Linear"`或`"MultiheadAttention"`。
        `head_mask_list`: 如果`module_type`为`"MultiheadAttention"`，则应为`pruned_module_list`中
        每一块的剪枝mask的列表，否则传入`None`或空列表`[]`

    Output:
        「并联」结构`MultipleParallelFinetuningBlock`。
    """
    if head_mask_list == []:
        head_mask_list = None
    # 先从原模型中找原块
    tokens = name_pruned.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    # 现在cur_mod是原块
    # 然后用 MultipleParallelFinetuningBlock 类制作并联块
    new_block = MultipleParallelFinetuningBlock(cur_mod, module_pruned_list, module_type, head_mask_list)     # 没有head_mask时，默认为None
    return new_block
