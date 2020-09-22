import os
import pandas as pd
import numpy as np
from utils import parse_args


def ensemble_in_model(version):
    model_path = f"./models{args.version}/"
    csv_files = [file for file in os.listdir(model_path) if ".csv" in file and file.startswith(version)]
    data = pd.read_csv(model_path + csv_files[0])
    for i in range(1, len(csv_files)):
        data['target'] += pd.read_csv(model_path + csv_files[i])['target']

    data['target'] /= len(csv_files)
    data.to_csv(f"{model_path}{version}_FINAL.csv", index=False)
    print(f"{version}_FINAL.csv saved")


def ensemble_models():
    csv_dirs = [file for file in os.listdir("./") if file.startswith('modelsE')]
    count = 0
    res = pd.read_csv('/home/xinsheng/skinImage/sample_submission.csv')
    tmp = res['target']

    for dir in csv_dirs:
        for file in os.listdir(dir):
            if "Fold_1.csv" in file:
                df = pd.read_csv(f"./{dir}/{file}")
                res['target'] += df["target"]
                count += 1
    res['target'] -= tmp
    res['target'] /= count
    res.to_csv("ENSEMBEL.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    # args.version = "ImageSize512"
    # version = 'EFFNET_WCRI_NOCAV_EXTRA'
    # ensemble_in_model(args.version)
    ensemble_models()
