# Composability in Transformer Pruning

**NOTE: Unfinished. Actively updating.**

A framework to utilize the composability between pruning configurations in the pruning of Transformer models. This implementation only focuses on the pruning of Linear (FC) layers and Multi-head Attention layers.

## Usage

```bash
# Execute each on need
python train_origin.py
python prune.py -r PRUNING_RATE [--generate_config]
python finetune.py -e EPOCHS -r PRUNING_RATE [--batch_size BATCH_SIZE]
python retrain.py -r PRUNING_RATE -f FINETUNED_EPOCHS -e RETRAINING_EPOCHS [--batch_size BATCH_SIZE]
python evaluate.py model_file
```

## Terminology

- **Origin**: An origin model without pruning.
- **Baseline**: A pruned and directly model-level finetuned model. No block-level finetuning.
- **Block-level Finetuned**: The resulting model from `finetune.py`.
- **Model-level Finetuned**: The resulting model from `retrained.py`.

## Modules

- `train_origin.py`: Trains a default Transformer that can be used in machine translation on Multi30k dataset, which contains Ger-Eng sentence pairs.
- `prune.py`: Block-level pruning. Will read config from `conf_prune.json`.
- `finetune.py`: Block-level finetuning.
- `retrain.py`: Model-level finetuning.
- `evaluate.py`: Evaluates a model based on BLEU score on Multi30k dataset.
