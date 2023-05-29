# Composability in Transformer Pruning

**NOTE: Unfinished. Actively updating.**

A framework to utilize the composability between pruning configurations in the pruning of Transformer models. This implementation only focuses on the pruning of Linear (FC) layers and Multi-head Attention layers.

## Usage

Execute each on need.

**Train a original model:**

```bash
python train_origin.py
```

**Generate a default configuration file:**

```bash
python config_generator.py -m MODEL_FILE [-n CONFIG_NUMBER -o CONFIG_FILENAME]
```

Modify that configuration file on need. (Default `conf.json` with `CONFIG_NUMBER=3` copies of rate `0.5`.)

**Generate pruning plan from configuration file:**

```bash
python config_prune_generator.py -i CONFIG_FILE
```

Do NOT modify that pruning plan file `conf_prune.json`, otherwise Model-level Finetuning would NOT work.

**Prune based on pruning plan:**

```bash
# python prune_single.py -r PRUNING_RATE [--generate_config]
python prune.py
```

**Block-level finetune:**

```bash
# python finetune_single.py -e EPOCHS -r PRUNING_RATE [--batch_size BATCH_SIZE]
python finetune.py -m MODULES_FILE -e BLOCK_FINETUNING_EPOCHS
```

**Model-level finetune (multi-model finetuning unfinished):**

```bash
# python retrain_single.py -r PRUNING_RATE -f BLOCK_FINETUNED_EPOCHS -e MODEL_FINETUNING_EPOCHS [--batch_size BATCH_SIZE]
python retrain.py -m FINETUNED_MODULES -o ORIGINAL_MODEL -c CONFIG_FILE -n CONFIG_# -e EPOCHS
```

`retrain.py` finetunes ONLY ONE configuration at each running. Specify which configuration to apply by `-n CONFIG_#` (starts at 0).

**Evaluate a model:**

```bash
python evaluate.py model_file
```

### Example

```bash
python train_origin.py
python config_generator.py -m ./model/baseline.pth
# Do some modification to conf.json, then continue
python config_prune_generator.py -i ./conf.json
python prune.py
python finetune.py -m ./numpy/modules_pruned_with_conf.npy -e 30
python retrain.py -m ./numpy/modules_finetuned_conf_epoch30.npy -o ./model/baseline.pth -c ./conf.json -n 1 -e 5
```

## Terminology

- **Origin**: An origin model without pruning.
- **Baseline**: A pruned and directly model-level finetuned model. No block-level finetuning.
- **Block-level Finetuned**: The resulting model from `finetune.py`.
- **Model-level Finetuned**: The resulting model from `retrained.py`.

## Modules

- `train_origin.py`: Trains a default Transformer that can be used in machine translation on Multi30k training set, which contains Ger-Eng sentence pairs.
- `prune.py`: Block-level pruning. Will read config from `conf_prune.json`.
- `finetune.py`: Block-level finetuning.
- `retrain.py`: Model-level finetuning.
- `evaluate.py`: Evaluates a model based on BLEU score on Multi30k validation set.
