#!/home/tongping/anaconda3/envs/transformer/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', type=str, help='Path to csv file')
args = parser.parse_args()

assert os.path.exists(args.csv_file), f'File {args.csv_file} does not exist'

df = pd.read_csv(args.csv_file)
# df heads: epoch,Training Loss,Valid. Loss,Training Time,Validation Time)
# plot Train & Validation loss and save to `loss.png`
plt.plot(df['epoch'], df['Training Loss'], label='Train Loss')
plt.plot(df['epoch'], df['Valid. Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png', dpi=300)

print('Plots saved to `loss.png`')
