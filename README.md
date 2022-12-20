# Composability in Transformer Pruning

How to use:

```bash

python train_origin.py

python prune.py -r PRUNING_RATE [--generate_config]

python finetune.py -e EPOCHS -r PRUNING_RATE [--batch_size BATCH_SIZE]

python retrain.py -r PRUNING_RATE -f FINETUNED_EPOCHS -e RETRAINING_EPOCHS [--batch_size BATCH_SIZE]

python evaluate.py model_file

```

Unfinished. Actively updating.
