import datetime
import os
import torch
import torch.nn as nn
import transformers.models.t5.modeling_t5 as t5


def set_module(model: nn.Module, submodule_key: str, module: nn.Module) -> None:
    # Replace a module of model with a new module by name.
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


def format_time(elapsed):
    # Format time
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def type_of_t5_module(name: str, module: nn.Module) -> str:
    """Return the type of a T5 module.
    An attention module ends with `.SelfAttention` or `.EncDecAttention`. A
    linear module in attention ends with `.o`. A linear module in FFN ends with
    `.wi` or `.wo`.

    Args:
        :param name: name of module
        :param module: module

    Returns:
        :return: type of module. "MultiheadAttention" or "Linear" or None.
    """
    # Attention. `.q`, `.k`, `.v`
    if isinstance(module, nn.Linear) and any([x in name for x in [".q", ".k", ".v"]]):
        return "MultiheadAttention"
    # Linear output project in Attention layer. `.o`
    elif isinstance(module, nn.Linear) and ".o" in name:
        return "Linear"
    # Linear in FFN. `.wi`, `.wo`
    elif isinstance(module, nn.Linear) and any([x in name for x in [".wi", ".wo"]]):
        return "Linear"
    else:
        return None


def save_checkpoint(model, optimizer, scheduler, this_epoch, loss, training_stats, early_stopping_best_loss, early_stopping_patience_counter, save_path):
    checkpoint = {
        'epoch': this_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'training_stats': training_stats,
        'early_stopping_best_loss': early_stopping_best_loss,
        'early_stopping_patience_counter': early_stopping_patience_counter
    }
    torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{this_epoch}.pt'))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    loss = checkpoint['loss']
    training_stats = checkpoint['training_stats']
    early_stopping_best_loss = checkpoint['early_stopping_best_loss']
    early_stopping_patience_counter = checkpoint['early_stopping_patience_counter']

    return model, optimizer, scheduler, epoch, loss, training_stats, early_stopping_best_loss, early_stopping_patience_counter
