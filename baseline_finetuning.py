import json
import os
import time
import pandas as pd
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils import format_time, save_checkpoint, load_checkpoint, type_of_model, get_embed_dim, get_num_heads


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="./model/t5-small_pruned.pth", help='model file')
parser.add_argument('-t', '--tokenizer', type=str, default="./tokenizer/t5-small.pth", help='tokenizer file')
parser.add_argument("--check_point", type=str, default="./model/checkpoint/baseline_finetune/", help="checkpoint directory")
parser.add_argument('-o', '--output', type=str, default="./model/t5-small_baseline_finetuned_unspecified.pth", help='output .pth')
parser.add_argument('--stats', type=str, default="./model/t5-small_baseline_finetuned_unspecified_stats.csv", help='output stats file')
args = parser.parse_args()

PATH_TO_ORIGINAL_MODEL = args.model     # input
PATH_TO_TOKENIZER = args.tokenizer      # input
assert os.path.exists(PATH_TO_ORIGINAL_MODEL), "Model file (.pth) not found!"
assert os.path.exists(PATH_TO_TOKENIZER), "Tokenizer file (.pth) not found!"
PATH_TO_OUTPUT_MODEL = args.output      # output
PATH_TO_OUTPUT_STATS = args.stats       # output
PATH_TO_CHECKPOINT = args.check_point   # checkpoint
if not os.path.exists(PATH_TO_CHECKPOINT):
    os.mkdir(PATH_TO_CHECKPOINT)

TYPE_OF_MODEL = type_of_model(PATH_TO_ORIGINAL_MODEL)
assert TYPE_OF_MODEL is not None, "Model type not recognized!"
print(f"Model type: {TYPE_OF_MODEL}")

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
MANUAL_SEED = 42
VALID_SET_SIZE = 1000
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
model = torch.load(PATH_TO_ORIGINAL_MODEL)

# Some other hyperparameters
EMB_SIZE = get_embed_dim(TYPE_OF_MODEL, model)
NHEAD = get_num_heads(TYPE_OF_MODEL, model)


# Load dataset
if TYPE_OF_MODEL == 't5':
    from dataset.t5 import T5Multi30kEnDe
    train_set = T5Multi30kEnDe(split='train')
    val_set = T5Multi30kEnDe(split='valid')
elif TYPE_OF_MODEL == 'distilbert':
    from datasets import load_dataset
    train_set = load_dataset('imdb', split='train')
    val_set = load_dataset('imdb', split='test')    # imdb does not have a validation set
    indices = np.random.choice(len(val_set), size=VALID_SET_SIZE, replace=False)
    val_set = val_set.select(indices.tolist())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


def tokenize_t5(batch):
    batch_src = batch['src']
    batch_tgt = batch['tgt']
    # token_en = tokenizer(padded_text_en, padding=True, truncation=True, max_length=768, return_tensors="pt")
    # token_de = tokenizer(padded_text_de, padding=True, truncation=True, max_length=768, return_tensors="pt")
    token_src = tokenizer(batch_src, padding=True, truncation=True, return_tensors="pt")
    token_tgt = tokenizer(batch_tgt, padding=True, truncation=True, return_tensors="pt")
    # Now, token_src/tgt is a dict with keys 'input_ids' and 'attention_mask'.

    return token_src, token_tgt


def tokenize_distilbert(batch):
    batch_text = batch['text']
    batch_label = batch['label']
    token_text = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt")

    return token_text, batch_label


def tokenize(MODEL_TYPE: str, batch):
    if MODEL_TYPE == 't5':
        return tokenize_t5(batch)
    elif MODEL_TYPE == 'distilbert':
        return tokenize_distilbert(batch)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not recognized!")


def train(model, scheduler, optimizer, criterion, num_epoch, trainloader, trainloader_length, valloader, valloader_length, continue_training=False, checkpoint_path=None):
    """
    Train the model.
    :param model: model to be trained
    :param scheduler: scheduler
    :param optimizer: optimizer
    :param criterion: loss function. DistilBERT-imdb used.
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
            source, target = tokenize(TYPE_OF_MODEL, batch)
            source, target = source.to(device), target.to(device)

            optimizer.zero_grad()

            if TYPE_OF_MODEL == 't5':
                output = model(
                    source['input_ids'],
                    labels=target['input_ids'],
                    # attention_mask=source['attention_mask']
                )
                loss = output.loss
            elif TYPE_OF_MODEL == 'distilbert':
                output = model(**source)
                # print(f'output.logits: {output.logits}')
                # print(f'target: {target}')
                loss = criterion(output.logits, target)

            batch_loss = loss.item()

            loss.backward()

            ###### TODO: gradient clipping ######

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
            source, target = tokenize(TYPE_OF_MODEL, batch)
            source, target = source.to(device), target.to(device)

            if TYPE_OF_MODEL == 't5':
                with torch.no_grad():
                    output = model(
                        source['input_ids'],
                        labels=target['input_ids'],
                        # attention_mask=source['attention_mask']
                    )
                    loss = output.loss
            elif TYPE_OF_MODEL == 'distilbert':
                with torch.no_grad():
                    output = model(**source)
                    loss = criterion(output.logits, target)

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


# PyTorch optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# HuggingFace optimizer:
# optimizer = AdamW(model.parameters(), lr=5e-4, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=1e2,
                                            num_training_steps=len(train_loader) * NUM_EPOCHS)

if TYPE_OF_MODEL == 'distilbert':
    # loss function for DistilBERT
    loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
model = model.to(device)
training_stats = train(
    model=model,
    scheduler=scheduler,
    optimizer=optimizer,
    criterion=loss_fn,
    num_epoch=NUM_EPOCHS,
    trainloader=train_loader,
    trainloader_length=len(train_loader),
    valloader=val_loader,
    valloader_length=len(val_loader),
    # continue_training=True,
    # checkpoint_path="./model/checkpoint/baseline_finetune/checkpoint_epoch_20.pt"
)

# Save the model
# PyTorch way
torch.save(model, PATH_TO_OUTPUT_MODEL)
print(f"Model saved to {PATH_TO_OUTPUT_MODEL}")
PATH_TO_SAVE_ARGS = "./model/checkpoint/baseline_finetune/args.json"
with open(PATH_TO_SAVE_ARGS, 'w') as f:
    json.dump(vars(args), f)
print(f"Args saved to {PATH_TO_SAVE_ARGS}")

# Save the training stats
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)
df_stats.to_csv(PATH_TO_OUTPUT_STATS)
