import json
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.t5 import T5Multi30kEnDe
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
from utils import format_time, set_module, save_checkpoint, load_checkpoint
from multiple_parallel_block import *
import argparse
import os


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--origin_model", type=str, default='./model/t5-small.pth', help="original (unpruned) model file (.pth)")
parser.add_argument('-t', '--tokenizer', type=str, default="./tokenizer/t5-small.pth", help='tokenizer file')
parser.add_argument("-p", "--pruned_modules", type=str, default='./numpy/module_pruned.npy', help="file of pruned modules")
parser.add_argument("--check_point", type=str, default="./model/checkpoint/block-level/", help="checkpoint directory")
parser.add_argument("-o", "--output", type=str, default="./numpy/module_finetuned.npy", help="output directory")
parser.add_argument("--stats", type=str, default="./model/t5-small_block-level_stats.csv", help="output stats file")
args = parser.parse_args()

PATH_TO_MODEL_ORIGIN = args.origin_model        # input
PATH_TO_MODULE_PRUNED = args.pruned_modules     # input
PATH_TO_TOKENIZER = args.tokenizer              # input
assert os.path.exists(PATH_TO_MODEL_ORIGIN), "Origin model directory not found!"
assert os.path.exists(PATH_TO_TOKENIZER), "Tokenizer file (.pth) not found!"
assert os.path.exists(PATH_TO_MODULE_PRUNED), "Pruned modules file not found!"
PATH_TO_OUTPUT = args.output                    # output
PATH_TO_OUTPUT_STATS = args.stats               # output
PATH_TO_CHECKPOINT = args.check_point           # checkpoint
if not os.path.exists(PATH_TO_CHECKPOINT):
    os.makedirs(PATH_TO_CHECKPOINT)

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE_SENTENCE = 256
MANUAL_SEED = 42
EARLY_STOPPING_PATIENCE = 3
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
model = torch.load(PATH_TO_MODEL_ORIGIN)
pruned_modules = np.load(PATH_TO_MODULE_PRUNED, allow_pickle=True).item()

# Some other hyperparameters
EMB_SIZE = model.config.d_model
NHEAD = model.config.num_heads

# Load dataset
train_set = T5Multi30kEnDe(split='train')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_SENTENCE, shuffle=False, num_workers=0, pin_memory=True)

val_set = T5Multi30kEnDe(split='valid')
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_SENTENCE, shuffle=False, num_workers=0, pin_memory=True)


def tokenize(batch):
    padded_text_src = batch['src']
    padded_text_tgt = batch['tgt']
    # token_en = tokenizer(padded_text_en, padding=True, truncation=True, max_length=768, return_tensors="pt")
    # token_de = tokenizer(padded_text_de, padding=True, truncation=True, max_length=768, return_tensors="pt")
    token_src = tokenizer.batch_encode_plus(padded_text_src, padding=True, truncation=True, return_tensors="pt")
    token_tgt = tokenizer.batch_encode_plus(padded_text_tgt, padding=True, truncation=True, return_tensors="pt")
    # Now, token_src/tgt is a dict with keys 'input_ids' and 'attention_mask'.

    return token_src, token_tgt


def train(model, scheduler, optimizer, criterion, num_epoch, trainloader, trainloader_length, valloader, valloader_length, continue_training=False, checkpoint_path=None):
    """
    Train the model.
    :param model: model to be trained
    :param scheduler: scheduler
    :param optimizer: optimizer
    :param criterion: loss function - pruned vs. original module, in parallel blocks
    :param num_epoch: number of epochs
    :param trainloader: trainloader
    :param trainloader_length: length of trainloader
    :param valloader: validation loader
    :param valloader_length: length of validation loader
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

            # Backward pass
            # Calculate Loss, and delete the pruned heads' gradients
            batch_loss = torch.tensor(0.0).to(device)
            for parallel_block in parallel_block_list:
                # The structure of parallel_block is: {module_type, original_module, pruned_module_list, out_original, out_pruned_list, [head_mask_list]}
                for i, module_pruned in enumerate(parallel_block.pruned_module_list):
                    loss = criterion(parallel_block.out_pruned_list[i], parallel_block.out_original)
                    batch_loss += loss
                    # The gradients have been generated. But pruned heads have also generated gradients.
                    # If the optimizer updates right now, the pruning will be invalid.
                    # Therefore, we set the gradients of these pruned heads to zero.
                    # First, determine whether this block is an MHA (Multi-head Attention).
                    # If it's a Linear layer, ignore it, because pytorch prune handles it.
                    if parallel_block.module_type == "MultiheadAttention":
                        num_heads = NHEAD
                        embed_dim = EMB_SIZE
                        head_dim = embed_dim // num_heads
                        head_mask = parallel_block.head_mask_list[i]
                        for head in range(num_heads):
                            dim_from = head * head_dim
                            dim_to = dim_from + head_dim
                            module_pruned.weight.grad.data[dim_from:dim_to].mul_(head_mask[head])

            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += batch_loss.item()

            # Update the progress bar
            progress_bar.set_postfix({'loss': f'{batch_loss.item():.3f}'})

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

            batch_loss = torch.tensor(0.0).to(device)
            with torch.no_grad():
                output = model(
                    source['input_ids'],
                    labels=target['input_ids'],
                    # attention_mask=source['attention_mask']
                )
                for parallel_block in parallel_block_list:
                    # The structure of parallel_block is: {module_type, original_module, pruned_module_list, out_original, out_pruned_list, [head_mask_list]}
                    for i, module_pruned in enumerate(parallel_block.pruned_module_list):
                        loss = criterion(parallel_block.out_pruned_list[i], parallel_block.out_original)
                        batch_loss += loss

            val_loss += batch_loss.item()

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


# Freeze all the parameters.
# The only parameters that will be trained are the parameters in
# the parallelblocks, which will be attached to the model later.
for param in model.parameters():
    param.requires_grad = False

param_to_finetune = []      # For optimizer
parallel_block_list = []    # For lossfn_block function. Keep a record of all the parallel blocks

for name_pruned in pruned_modules:
    # module_pruned_info_list is a list of {"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask (if MHA)}
    module_pruned_info_list = pruned_modules[name_pruned]
    module_type = ""        # "Linear" or "MultiheadAttention"
    module_pruned_list = []
    head_mask_list = []
    # for each module being pruned, construct a parallel block
    for module_pruned_info in module_pruned_info_list:
        layer = module_pruned_info["layer"]
        module_pruned_list.append(layer)
        # If it is a MHA, it has a head_mask. If it is a Linear, it doesn't.
        if hasattr(module_pruned_info, "head_mask"):
            if module_type == "":
                module_type = "MultiheadAttention"
            head_mask = module_pruned_info["head_mask"]
            head_mask_list.append(head_mask)
        else:
            if module_type == "":
                module_type = "Linear"
    # construct parallel block
    parallel_block = None
    if module_type == "MultiheadAttention":
        parallel_block = construct_parallel_block(model, name_pruned, module_pruned_list, "MultiheadAttention", head_mask_list)
    elif module_type == "Linear":
        parallel_block = construct_parallel_block(model, name_pruned, module_pruned_list, "Linear")
    # replace the original module with the parallel block
    set_module(model, name_pruned, parallel_block)
    # optimizer will only optimize the parameters in `parallel_block.pruned_module_list`
    for module in parallel_block.pruned_module_list:
        param_to_finetune.append({"params": module.parameters()})
    parallel_block_list.append(parallel_block)


# HuggingFace optimizer
optimizer = torch.optim.Adam(param_to_finetune, lr=5e-4)
# optimizer = AdamW(param_to_finetune, lr=5e-4, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=1e2,
                                            num_training_steps=len(train_loader))
lossfn_block = nn.MSELoss()     # loss function between each parallel block and the original block

# Train the model
model = model.to(device)
training_stats = train(
    model=model,
    scheduler=scheduler,
    optimizer=optimizer,
    criterion=lossfn_block,
    num_epoch=NUM_EPOCHS,
    trainloader=train_loader,
    trainloader_length=len(train_loader),
    valloader=val_loader,
    valloader_length=len(val_loader),
    # continue_training=True,
    # checkpoint_path="./model/checkpoint/block-level/checkpoint_epoch_5.pt"
)

# Unfreeze the model
for param in model.parameters():
    param.requires_grad = True

# Save the modules
np.save(PATH_TO_OUTPUT, pruned_modules, allow_pickle=True)
print(f"Saved the finetuned modules to {PATH_TO_OUTPUT}")
PATH_TO_SAVE_ARGS = "./model/checkpoint/block-level/args.json"
with open(PATH_TO_SAVE_ARGS, 'w') as f:
    json.dump(vars(args), f)
print(f"Saved the args to {PATH_TO_SAVE_ARGS}")

# Save the training stats
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)
df_stats.to_csv(PATH_TO_OUTPUT_STATS)
