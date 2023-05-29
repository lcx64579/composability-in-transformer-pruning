import torch
import torch.nn as nn


class MultipleParallelFinetuningBlock(nn.Module):
    r"""
    Construct a structure in which the original module is connected in parallel
    with multiple pruning modules, where the output of the original module is used
    as the output of the structure.

    Args:
        `original_module`: The original module
        `pruned_module_list`: A list of pruned modules
        `module_type`: The type of the module. Must be `"Linear"` or `"MultiheadAttention"`
        `head_mask_list`: If `module_type` is `"MultiheadAttention"`, then it should be
        a list of head masks of each module in `pruned_module_list`. Otherwise this
        attribute does not exist.

    Attributes:
        `original_module`: The original module
        `pruned_module_list`: A list of pruned modules
        `out_original`: The output of the original module. Used for loss calculation.
        `out_pruned_list`: A list of outputs of pruned modules. Used for loss calculation.
        `module_type`: The type of the module. Must be `"Linear"` or `"MultiheadAttention"`
        `head_mask_list`: If `module_type` is `"MultiheadAttention"`, then it should be
        a list of head masks of each module in `pruned_module_list`. Otherwise this
        attribute does not exist.

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
            self.head_mask_list = head_mask_list       # Only to keep a record. Not used in forward.
        # Things to keep t5's sanity:
        self.weight = self.original_module.weight

    def forward(self, *input, **kwargs):
        device = self.original_module.weight.device
        z_orig = self.original_module(*input, **kwargs)
        self.out_original = z_orig      # WARNING: Always be aware if z_orig is a tuple or not. In this case, it is just a Tensor.
        self.out_pruned_list = []           # Clear the list. Otherwise, the list will keep growing.
        for pruned_module in self.pruned_module_list:
            pruned_module.to(device)
            z_pruned = pruned_module(*input, **kwargs)
            self.out_pruned_list.append(z_pruned)
        return z_orig


def construct_parallel_block(model: nn.Module, name_pruned: str, module_pruned_list: list, module_type: str, head_mask_list: list = None) -> nn.Module:
    r"""
    Construct a structure in which the original module is connected in parallel
    with multiple pruning modules, where the output of the original module is used
    as the output of the structure.

    Args:
        :param model: The original model
        :param name_pruned: The name of the original module
        :param module_pruned_list: A list of pruned modules
        :param module_type: The type of the module. Must be `"Linear"` or `"MultiheadAttention"`
        :param head_mask_list: If `module_type` is `"MultiheadAttention"`, then it should be a list of head masks of each module in `pruned_module_list`. Otherwise it should be `None`.

    Output:
        The parallel structure module `MultipleParallelFinetuningBlock`
    """
    if head_mask_list == []:
        head_mask_list = None
    # Find the original module
    tokens = name_pruned.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    # Now, cur_mod is the original module
    # Construct the new block
    new_block = MultipleParallelFinetuningBlock(cur_mod, module_pruned_list, module_type, head_mask_list)     # 没有head_mask时，默认为None
    return new_block
