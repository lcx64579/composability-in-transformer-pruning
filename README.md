# Composability in Transformer Pruning

**NOTE: Unfinished. Actively updating.**

## Introduction

This is my master's thesis project.

Sometimes, you might want to prune and fine-tune a model into several different versions, and these versions differ only in some submodules. Here's the idea: for those common modules, we fine-tune them only once, and then combine them into each complete model. In this way, each model can now start fine-tuning at a better initial parameter setting, and the fine-tuning time for all models can be reduced. We call it the **composability**.

This project is a framework that leverages the composability between pruning configurations in the pruning & fine-tuning process of Transformer models to speed up the process. The focus of this implementation is on the pruning of Linear (both feed forward and attention out-projection) modules, as well as Multi-head Attention modules. By default, it applies $l_1$-unstructured pruning to Linear weight matrices, and head-wise $l_1$-structured pruning to Attention QKV matrices.

Explanation and thesis will be added finally.

> **Terminology**
>
> Using term **Block-level** in code instead of **Module-level** because it is more readable. In fact **Module-level** is the more appropriate expression.
>
> The terms **layer** and **module** are interchangeable in this project.
>
> **Notice**
>
> In most of the codes I set the device to `cuda:1` because I have two GPUs and I want to use the second one. If you have only one GPU, change it to `cuda`.
>
> A Seq2Seq model is included in the source files for my convenience; However, for the full usage of it, please see `./legacy-seq2seq/` folder.

## Dependencies

Huggingface's transformers, PyTorch, and torchtext are the basic requirements. Other dependencies such as numpy, matplotlib, etc. are also needed.

## Usage

Most of the arguments are optional.

**Download example model `t5-small`:**

```bash
python download_model.py MODEL_TYPE
```

Downloads a model from Huggingface to `./model/`, together with its tokenizer to `./tokenizer/`.

`MODEL_TYPE` Options:

- `t5`: T5-small 60M params model for Machine Translation
- `distilbert`: DistilBERT-imdb model for Sentiment Analysis

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
python prune.py -m MODEL -c CONF -o OUTPUT_MODULES
```

`CONF` defaults to `conf_prune.json`.

**Compose a model with pruned/finetuned modules based on a configuration:**

```bash
python compose_model.py -m ORIGIN_MODEL -t TOKENIZER -p MODULES -c CONF_FILE -n USE_NTH_CONFIG -o OUTPUT_MODEL
```

**Finetune a baseline model:**

```bash
python model-level_finetuning.py -m ORIGIN_MODEL -p PRUNED_MODULES -c CONF_FILE -n USE_NTH_CONFIG -o OUTPUT_PTH --stats TRAINING_STATS_CSV
```

Finetunes a composed pruned model directly, without module-level finetuning. This is the baseline used for the control experiment.

This is basically the same command as **Model-level Finetuning**. Only replace the `-p FINETUNED_MODULES` parameter with the pruned modules without finetuning `-p PRUNED_MODULES`.

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
python evaluate-t5.py -m MODEL_FILE -t TOKENIZER
# or
python evaluate-distilbert.py -m MODEL_FILE -t TOKENIZER
```

Evaluates the model and reports the performance.

Matrices:

- `t5`: BLEU score (0~100) on Multi30K test set.
- `distilbert`: accuracy score (0~1) on IMDb test set.

**Plot training loss and validation loss:**

```bash
./plot_loss.py TRAINING_STATS_CSV
```

Plots losses and saves to `loss.png`.

## Experiment Setup

**T5-small**:

- Model: [T5 small 60M parameters](https://huggingface.co/t5-small)
- Dataset: [Multi30K](https://pytorch.org/text/stable/_modules/torchtext/datasets/multi30k.html)

**DistilBERT**:

> Not recommended. With learning rate = 5e-5 ~ 1e-3, DistilBERT-imdb overfits on IMDb dataset with more than 1 epoch (`lr=1e-3, batch_size=128`), even after pruning.
>
> If you really want to do it, try a learning rate that is REALLY small, such as `lr=1e-6`.

- Model: [DistilBERT-imdb](https://huggingface.co/lvwerra/distilbert-imdb)
- Dataset: [IMDb](https://huggingface.co/datasets/imdb)
