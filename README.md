# Composability in Transformer Pruning

**NOTE: Unfinished. Actively updating.**

## Introduction

This is my master's thesis project.

A framework to utilize the composability between pruning configurations in the pruning of Transformer models. This implementation only focuses on the pruning of Linear (feed forward & attention out-projection) modules and Multi-head Attention modules. By default it applies $l_1$-unstructured pruning on Linear weight matrices, head-wise $l_1$-structured pruning on Attention QKV matrices.

Explanation and thesis will be added finally.

- Model: [T5 small 60M parameters](https://huggingface.co/t5-small)
- Dataset: [Multi30K](https://pytorch.org/text/stable/_modules/torchtext/datasets/multi30k.html)

> **Terminology**
>
> Using term **Block-level** in code instead of **Module-level** because it is more readable. In fact **Module-level** is the more appropriate expression.
>
> The terms **layer** and **module** are interchangeable in this project.

## Dependencies

You need at least Huggingface's `transformers`, PyTorch and `torchtext`. Other dependencies such as numpy, matplotlib, etc. are also needed.

## Usage

Most of the arguments are optional.

**Download example model `t5-small`:**

```bash
python download_model.py
```

By default downloads a T5-small 60M params model from Huggingface.

**Generate a default configuration file:**

```bash
python config_generator.py -m MODEL -o OUTPUT_CONF_FILE_JSON --attention PRUNING_RATE_1 PRUNING_RATE_2 ... --linear PRUNING_RATE_1 PRUNING_RATE_2 ... -n TOTAL_CONF_NUMBER
```

Default output filename `conf.json`.

`--attention` and `--linear` take lists of number as pruning rates. List length should be identical to `TOTAL_CONF_NUMBER`. For example, if we pass `--attention 0.25 0.5 0.25 --linear 0.3 0.3 0.5 -n 3`, the 3 configs will be (attention: 0.25, linear 0.3), (attention 0.5, linear 0.3), (attention 0.25, linear 0.5).

It's a default template. All these rates can be adjusted in need in `OUTPUT_CONF_FILE_JSON` file.

**Generate pruning scheme from configuration file:**

```bash
python prune_scheme_generator.py -i CONF_FILE_JSON -o OUTPUT_JSON
```

*Do NOT modify that pruning scheme file* (default `conf_prune.json`), otherwise model-level Finetuning would NOT work. If you don't like your scheme, adjust configuration file (default `conf.json`) and regenerate pruning scheme instead.

**Prune based on pruning scheme:**

```bash
python prune.py -m MODEL -c CONF -o OUTPUT_MODULES -p OUTPUT_PRUNED_MODEL
```

`OUTPUT_PRUNED_MODEL` is a test model pruned with every first module in pruning scheme. It's not necessary.

**Compose a model with pruned/finetuned modules based on a configuration:**

```bash
python compose_model.py -m ORIGIN_MODEL -p MODULES -c CONF_FILE -n USE_NTH_CONFIG -o OUTPUT_MODEL
```

**Finetune a baseline model:**

```bash
python baseline_finetuning.py -m MODEL -o OUTPUT_MODEL --stats TRAINING_STATS_CSV
```

Finetunes a composed pruned model directly, without module-level finetuning. This is the baseline used for the control experiment.

**Module(Block)-level finetune:**

```bash
python block-level_finetuning.py -m ORIGIN_MODEL -p PRUNED_MODULES -o OUTPUT_MODULES --stats TRAINING_STATS_CSV
```

Finetunes each pruned modules in the pruning scheme, and save them as a numpy object.

**Model-level finetune:**

```bash
python model-level_finetuning.py -m ORIGIN_MODEL -p FINETUNED_MODULES -c CONF_FILE -n USE_NTH_CONFIG -o OUTPUT_PTH --stats TRAINING_STATS_CSV
```

`model-level_finetuning.py` finetunes ONLY ONE configuration at each running. Specify which configuration to apply by `-n USE_NTH_CONFIG` (starts at 0).

**Evaluate a model:**

```bash
python evaluate.py -m model_file
```

Evaluates the model and reports the BLEU score on Multi30K dataset.

**Plot training loss and validation loss:**

```bash
./plot_loss.py TRAINING_STATS_CSV
```

Plots losses and saves to `loss.png`.
