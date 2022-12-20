import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str)
args = parser.parse_args()
MODEL_FILE = "./models/" + args.model_file
print(MODEL_FILE)