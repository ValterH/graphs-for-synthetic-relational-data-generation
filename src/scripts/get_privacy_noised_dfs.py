import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tqdm
import os
import re

from src.evaluation.eval_privacy import get_args, calculate_privacy, drop_ids
from src.evaluation.eval_classifier import CustomHyperTransformer
from src.data.utils import load_tables, load_metadata, get_root_table, merge_children
from src.baselines.noiser import generate_noised_dataset

ROOT_PATH = "data/synthetic/noiser"

def prepare_noised(args, noise_level_num, noise_level_cat):
    tables_synthetic = generate_noised_dataset(args.dataset, 
                                               save_=False, 
                                               noise_level_num=noise_level_num,
                                               noise_level_cat=noise_level_cat)
    tables_original = load_tables(args.dataset, split='train')

    metadata = load_metadata(args.dataset)
    root_table = get_root_table(args.dataset)

    original = merge_children(tables_original, metadata, root_table)
    synthetic = merge_children(tables_synthetic, metadata, root_table)

    # drop all foreign and primary keys
    for table in metadata.to_dict()['tables'].keys():
        drop_ids(original, table, metadata)
        drop_ids(synthetic, table, metadata)

    transformed_original = original.copy()
    transformed_synthetic = synthetic.copy()

    column_names = transformed_original.columns.to_list()
    transformed_original = transformed_original.reindex(column_names, axis=1)
    transformed_synthetic = transformed_synthetic.reindex(column_names, axis=1)

    max_items = 100000

    n = min(max_items, transformed_original.shape[0], transformed_synthetic.shape[0])
    transformed_original = transformed_original.sample(n=n, random_state=42, replace=False)
    transformed_synthetic = transformed_synthetic.sample(n=n, random_state=42, replace=False)

    if 'Date' in column_names:
        transformed_original['Date'] = pd.to_numeric(pd.to_datetime(transformed_original['Date']))
        transformed_synthetic['Date'] = pd.to_numeric(pd.to_datetime(transformed_synthetic['Date']))

    ht = CustomHyperTransformer()
    transformed_original = ht.fit_transform(transformed_original)
    transformed_synthetic = ht.fit_transform(transformed_synthetic)

    return transformed_original, transformed_synthetic


if __name__ == "__main__":
    
    args = get_args()
    folder = os.path.join(ROOT_PATH, args.dataset, "merged")

    transformed_original = pd.read_csv(os.path.join(folder, "original.csv"))

    results = []
    noises = []
    for file in os.listdir(os.path.join(ROOT_PATH, args.dataset, "merged")):
        transformed_synthetic = pd.read_csv(os.path.join(folder, file))
        # in case of new category columns we need to do this:
        cols = list(set(transformed_synthetic.columns) & set(transformed_original.columns))

        privacy_score = calculate_privacy(transformed_original[cols], transformed_synthetic[cols], privacy_percentile=args.privacy_percentile)

        if file.split(".")[0] == "original":
            noise = 0.0
        else:
            noise = float(re.search(r"(\d+\.\d+)", file).group(1))

        results.append(privacy_score)
        noises.append(noise)

    df = pd.DataFrame()
    df["noise"] = noises
    df["privacy"] = results

    
    df.to_csv("eval/privacy/noiser/privacy_vs_noise.csv")




