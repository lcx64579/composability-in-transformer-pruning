import json
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from dataset.t5 import T5Multi30kEnDe
from torch.utils.data import DataLoader
from utils import format_time, set_module, save_checkpoint, load_checkpoint


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--origin_model", type=str, default='./model/t5-small.pth', help="original (unpruned) model file (.pth)")
parser.add_argument('-t', '--tokenizer', type=str, default="./tokenizer/t5-small.pth", help='tokenizer file')
parser.add_argument("-p", "--finetuned_modules", type=str, default='./numpy/module_finetuned.npy', help="file of finetuned modules")
parser.add_argument("-c", "--conf_file", type=str, default='./conf.json', help="configuration file")
parser.add_argument('-n', '--finetune_nth_config', type=int, default=0, help="This program finetunes ONLY ONE config at each running. Specify the `n`-th config in conf.json to be worked on. `n` starts at 0.")
parser.add_argument("--check_point", type=str, default="./model/checkpoint/model-level/", help="checkpoint directory")
parser.add_argument('-o', '--output', type=str, default="./model/t5-small_model-level.pth", help='output .pth')
parser.add_argument('--stats', type=str, default="./model/t5-small_model-level_stats.csv", help='output stats file')
args = parser.parse_args()

PATH_TO_ORIGINAL_MODEL = args.origin_model          # input
PATH_TO_TOKENIZER = args.tokenizer                  # input
PATH_TO_FINETUNED_MODULES = args.finetuned_modules  # input
PATH_TO_CONFIG = args.conf_file                     # input
assert os.path.exists(PATH_TO_ORIGINAL_MODEL), "Model file (.pth) not found!"
assert os.path.exists(PATH_TO_TOKENIZER), "Tokenizer file (.pth) not found!"
assert os.path.exists(PATH_TO_FINETUNED_MODULES), "Finetuned modules file (.npy) not found!"
assert os.path.exists(PATH_TO_CONFIG), "Configuration file (.json) not found!"
PATH_TO_OUTPUT_MODEL = args.output      # output
PATH_TO_OUTPUT_STATS = args.stats       # output
PATH_TO_CHECKPOINT = args.check_point   # checkpoint
if not os.path.exists(PATH_TO_CHECKPOINT):
    os.mkdir(PATH_TO_CHECKPOINT)

NTH_CONFIG = args.finetune_nth_config

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 256
MANUAL_SEED = 42
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_SAVE_EVERY = 5

# Set random seed
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load everything
tokenizer = torch.load(PATH_TO_TOKENIZER)
model = torch.load(PATH_TO_ORIGINAL_MODEL)
finetuned_modules = np.load(PATH_TO_FINETUNED_MODULES, allow_pickle=True).item()
conf_file = open(PATH_TO_CONFIG, 'r')
conf = json.load(conf_file)
model_total_number = len(conf)


# Some other hyperparameters
EMB_SIZE = model.config.d_model
NHEAD = model.config.num_heads

# Load dataset
train_set = T5Multi30kEnDe('train')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

val_set = T5Multi30kEnDe('valid')
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


def tokenize(batch):
    padded_text_src = batch['src']
    padded_text_tgt = batch['tgt']
    # token_en = tokenizer(padded_text_en, padding=True, truncation=True, max_length=768, return_tensors="pt")
    # token_de = tokenizer(padded_text_de, padding=True, truncation=True, max_length=768, return_tensors="pt")
    token_src = tokenizer.batch_encode_plus(padded_text_src, padding=True, truncation=True, return_tensors="pt")
    token_tgt = tokenizer.batch_encode_plus(padded_text_tgt, padding=True, truncation=True, return_tensors="pt")
    # Now, token_src/tgt is a dict with keys 'input_ids' and 'attention_mask'.

    return token_src, token_tgt


def find_finetuned_module_by_name_and_ratio(finetuned_modules: dict, name: str, ratio: float) -> tuple[nn.Module, list]:
    assert name in finetuned_modules, f"MODULE {name} NOT FOUND"
    info_module_list = finetuned_modules[name]      # [{"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask}, ...]
    for info_module in info_module_list:            # {"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask}
        if info_module["ratio"] == ratio:
            if hasattr(info_module, "head_mask"):       # is a MultiheadAttention layer
                return info_module["layer"], info_module["head_mask"]
            else:
                return info_module["layer"], None
    return None, None


def train(model, scheduler, optimizer, num_epoch, trainloader, trainloader_length, valloader, valloader_length, multihead_attn_modules, continue_training=False, checkpoint_path=None):
    """
    Train the model.
    :param model: model to be trained
    :param scheduler: scheduler
    :param optimizer: optimizer
    :param num_epoch: number of epochs
    :param trainloader: trainloader
    :param trainloader_length: length of trainloader
    :param valloader: validation loader
    :param valloader_length: length of validation loader
    :param multihead_attn_modules: list of MultiheadAttention modules that should clear the gradient
    :param continue_training: whether to continue training from a checkpoint
    :param checkpoint_path: path to the checkpoint. if `continue_training`, this is required
    :return: Training stats.
    """
    start_epoch = 0
    early_stopping_best_loss = float('inf')
    early_stopping_patience_counter = 0

    training_stats = []       # Record training stats. Format: {'epoch', 'Training Loss', 'Valid. Loss', 'Training Time', 'Validation Time'}

    # Load checkpoint if continue_training is True
    if continue_training:
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."
        model, optimizer, scheduler, start_epoch, _, training_stats, early_stopping_best_loss, early_stopping_patience_counter = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")

    print("Start training...")
    time_total_start = time.time()

    for epoch in range(start_epoch, num_epoch):
        time_epoch_start = time.time()
        running_loss = 0.0
        # Wrap the iterator with tqdm
        progress_bar = tqdm(enumerate(trainloader, 0), total=trainloader_length, desc=f"Epoch {epoch + 1}/{num_epoch}")

        model.train()
        for i, batch in progress_bar:
            source, target = tokenize(batch)
            source, target = source.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(
                source['input_ids'],
                labels=target['input_ids'],
                # attention_mask=source['attention_mask']
            )

            loss = output.loss
            batch_loss = loss.item()

            loss.backward()

            # Make sure MultiheadAttention layer pruning is still there, by setting corresponding grad to 0
            for module in multihead_attn_modules:
                num_heads = NHEAD
                embed_dim = EMB_SIZE
                head_dim = embed_dim // num_heads
                head_mask = module.head_mask
                for head in range(num_heads):
                    dim_from = head * head_dim
                    dim_to = dim_from + head_dim
                    module.weight.grad.data[dim_from:dim_to].mul_(head_mask[head])


            optimizer.step()
            scheduler.step()

            running_loss += batch_loss
            # Update the progress bar
            progress_bar.set_postfix({'loss': f'{batch_loss:.3f}'})

        training_time = format_time(time.time() - time_epoch_start)

        avg_loss = running_loss / trainloader_length

        print(f'Epoch:{epoch+1:2}/{num_epoch}, Avg Loss: {avg_loss}, Time: {training_time}')

        # Validation
        print("Start validation...", end='')
        time_val_start = time.time()
        model.eval()
        val_loss = 0.0
        for batch in valloader:
            source, target = tokenize(batch)
            source, target = source.to(device), target.to(device)

            with torch.no_grad():
                output = model(
                    source['input_ids'],
                    labels=target['input_ids'],
                    # attention_mask=source['attention_mask']
                )
                loss = output.loss

            batch_loss = loss.item()
            val_loss += batch_loss

        val_time = format_time(time.time() - time_val_start)
        avg_val_loss = val_loss / valloader_length
        print(f'\rAvg Val Loss: {avg_val_loss}, Val Time: {val_time}')

        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': val_time
            }
        )

        # Early stopping
        if avg_val_loss < early_stopping_best_loss:
            early_stopping_best_loss = avg_val_loss
            early_stopping_patience_counter = 0
        elif avg_val_loss >= early_stopping_best_loss:
            early_stopping_patience_counter += 1
            if early_stopping_patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_SAVE_EVERY == 0:
            print(f'Saving checkpoint at epoch {epoch+1}')
            save_checkpoint(model, optimizer, scheduler, epoch+1, running_loss, training_stats, early_stopping_best_loss, early_stopping_patience_counter, PATH_TO_CHECKPOINT)

    time_total = format_time(time.time() - time_total_start)
    print(f'Finished Training. Total Time: {time_total}')

    return training_stats


conf_nth = conf[NTH_CONFIG]
print(f"Finetuning the {NTH_CONFIG}-th config in {PATH_TO_CONFIG}...")

multihead_attn_modules = []     # Save the MultiHeadAttention modules for later grad zero
for name in conf_nth:
    ratio = conf_nth[name]
    if ratio == 1:      # Skip if ratio is 1, which is the original version of module
        continue
    assert name in finetuned_modules, f"Module {name} not found in finetuned_modules"
    module, head_mask = find_finetuned_module_by_name_and_ratio(finetuned_modules, name, ratio)
    assert module is not None, f"Module {name} by ratio {ratio} is not found in finetuned_modules"
    if head_mask is not None:   # If not None, it is a MultiHeadAttention module
        assert not hasattr(module, 'head_mask'), f"Module {name} already has head_mask"
        module.head_mask = head_mask
        multihead_attn_modules.append(module)       # Save the module for later grad zero
    set_module(model, name, module)


# PyTorch optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# HuggingFace optimizer:
# optimizer = AdamW(model.parameters(), lr=5e-4, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=1e2,
                                            num_training_steps=len(train_loader) * NUM_EPOCHS)

# Train the model
model = model.to(device)


# Save an untrained version of the model
save_checkpoint(model, optimizer, scheduler, 0, float('inf'), [], float('inf'), 0, PATH_TO_CHECKPOINT)


training_stats = train(
    model=model,
    scheduler=scheduler,
    optimizer=optimizer,
    num_epoch=NUM_EPOCHS,
    trainloader=train_loader,
    trainloader_length=len(train_loader),
    valloader=val_loader,
    valloader_length=len(val_loader),
    multihead_attn_modules = multihead_attn_modules,
    # continue_training=True,
    # checkpoint_path="./model/checkpoint/model-level/checkpoint_epoch_5.pt"
)

# Save the model
# PyTorch way
torch.save(model, PATH_TO_OUTPUT_MODEL)
print(f"Model saved to {PATH_TO_OUTPUT_MODEL}")
PATH_TO_SAVE_ARGS = "./model/checkpoint/model-level/args.json"
with open(PATH_TO_SAVE_ARGS, 'w') as f:
    json.dump(vars(args), f)
print(f"Args saved to {PATH_TO_SAVE_ARGS}")

# Save the training stats
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)
df_stats.to_csv(PATH_TO_OUTPUT_STATS)
