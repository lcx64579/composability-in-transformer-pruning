import datetime
import os
import torch
import torch.nn as nn
import transformers.models.t5.modeling_t5 as t5


def set_module(model: nn.Module, submodule_key: str, module: nn.Module) -> None:
    r"""
    Replace a submodule in a module by a new module.

    Args:
        :param model: model
        :param submodule_key: a string of submodule key, e.g. "encoder.block.0.layer"
        :param module: new module

    Output:
        None
    """
    # referenced the implementation of `torch.quantization.fuse_modules()`
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
    An attention module ends with `.q`, `.k` or `.v`. A linear module in attention
    ends with `.o`. A linear module in FFN ends with `.wi` or `.wo`.

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


def type_of_distilbert_module(name: str, module: nn.Module) -> str:
    """Return the type of a DistilBERT module.
    An attention module ends with `.q_lin`, `.k_lin` or `.v_lin`. A linear module
    in attention ends with `.out_lin`. A linear module in FFN ends with `.lin1`
    or `.lin2`.
    """
    # Attention. `.q_lin`, `.k_lin`, `.v_lin`
    if isinstance(module, nn.Linear) and any([x in name for x in [".q_lin", ".k_lin", ".v_lin"]]):
        return "MultiheadAttention"
    # Linear output project in Attention layer. `.out_lin`
    elif isinstance(module, nn.Linear) and ".out_lin" in name:
        return "Linear"
    # Linear in FFN. `.lin1`, `.lin2`
    elif isinstance(module, nn.Linear) and any([x in name for x in [".lin1", ".lin2"]]):
        return "Linear"
    else:
        return None


def type_of_module(model_type: str, name: str, module: nn.Module) -> str:
    if model_type == "t5":
        return type_of_t5_module(name, module)
    elif model_type == "distilbert":
        return type_of_distilbert_module(name, module)
    else:
        # Do NOT raise error, as modules that are not Attention or Linear could be ignored.
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
