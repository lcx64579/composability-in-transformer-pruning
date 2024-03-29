import datetime
import os
import torch
import torch.nn as nn
import transformers.models.t5.modeling_t5 as t5
import transformers.models.distilbert.modeling_distilbert as distilbert


def set_module(model: nn.Module, submodule_key: str, module: nn.Module) -> None:
    """Replace a submodule in a module by a new module.

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


# Modify this to add supports to more model types.
def type_of_model(path_to_model: str) -> str:
    """Return the type of a model based on the path. **This requires the filename
    of the model indicates the type of it.**

    Args:
        :param path_to_model: path to model
    Returns:
        :return: type of model. None if not supported.
    """
    if 't5' in path_to_model:
        return 't5'
    elif 'distilbert' in path_to_model:
        return 'distilbert'
    else:
        return None


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
    if isinstance(module, t5.T5Attention):
        return "MultiheadAttention"
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
    if isinstance(module, distilbert.MultiHeadSelfAttention):
        return "MultiheadAttention"
    # Linear in FFN. `.lin1`, `.lin2`
    elif isinstance(module, nn.Linear) and any([x in name for x in [".lin1", ".lin2"]]):
        return "Linear"
    else:
        return None


# Modify this to add supports to more model types.
def type_of_module(model_type: str, name: str, module: nn.Module) -> str:
    if model_type == "t5":
        return type_of_t5_module(name, module)
    elif model_type == "distilbert":
        return type_of_distilbert_module(name, module)
    else:
        # Do NOT raise error, as modules that are not Attention or Linear could be ignored.
        return None


# Modify this to add supports to more model types.
def get_embed_dim(model_type: str, model: nn.Module) -> int:
    """Get the embedding dimension of a model.

    Args:
        :param model_type: type of model
        :param model: model

    Returns:
        :return: embedding dimension
    """
    if model_type == "t5":
        return model.config.d_model
    elif model_type == "distilbert":
        return model.config.dim
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")


# Modify this to add supports to more model types.
def get_num_heads(model_type: str, model: nn.Module) -> int:
    """Get the number of attention heads in a Multihead Attention layer of a model.

    Args:
        :param model_type: type of model
        :param model: model

    Returns:
        :return: number of heads
    """
    if model_type == "t5":
        return model.config.num_heads
    elif model_type == "distilbert":
        return model.config.n_heads
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")
