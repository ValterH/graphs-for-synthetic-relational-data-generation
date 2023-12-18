import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

from src.data.utils import merge_children, load_tables, load_metadata, get_root_table
from src.evaluation.eval_classifier import CustomHyperTransformer, drop_ids

# PRIVACY
def distance_to_closest_record(original, synthetic, **kwargs):
    """
    Calculate the distance to the closest record.
    """
    distances = pairwise_distances(original, synthetic, metric='manhattan')
    return np.min(distances, axis=1)


def nearest_neighbour_distance_ratio(original, synthetic, **kwargs):
    """
    Calculate the distance to the closest record.
    """
    distances = pairwise_distances(original, synthetic, metric='manhattan', **kwargs)
    distances = np.sort(distances, axis=1)
    nearest = distances[:, 0]
    second_nearest = distances[:, 1]
    return nearest / (second_nearest + 1e-10)

# graphVAE privacy (https://arxiv.org/pdf/2211.16889.pdf)
def nearest_neighbor_distribution_score(original, synthetic, privacy_percentile=0.05):
    """
    Calculate graphVAE's privacy score
    """
    original_numpy = original.copy().to_numpy()
    synthetic_numpy = synthetic.copy().to_numpy()

    # split original dataset in half for alpha calculation 
    D_1, D_2 = train_test_split(original_numpy, test_size=0.5)

    # calculate the 5th/x-th percentile 
    ratios = calculate_ratios(D_1, D_2)
    alpha = np.percentile(ratios, 100*privacy_percentile)

    indices = np.random.choice(original_numpy.shape[0], size=original_numpy.shape[0] // 2, replace=False)

    sampled_original = original_numpy[indices, :]
    sampled_synthetic = synthetic_numpy[indices, :]
    # calculate the ratios with synthetic data
    ratios_syn = calculate_ratios(sampled_original, sampled_synthetic)

    privacy = np.mean(ratios_syn < alpha)

    return privacy


def calculate_ratios(D_1, D_2):
    # calculate nearest neighbor distances for D_1 with itself and D_1 with D_2
    self_distance = pairwise_distances(D_1, metric="manhattan")
    self_distance = np.partition(self_distance, 1, axis=1)[:, 1] # np.min is distance to itself so we need to take second min
    mutual_distance = pairwise_distances(D_1, D_2, metric="manhattan")
    mutual_distance = np.min(mutual_distance, axis=1)

    ratios = mutual_distance / (self_distance)
    
    return ratios


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='LP and lep pozdrav')

    # dataset to evaluate
    parser.add_argument('--dataset', type=str, default='rossmann-store-sales',
                        help='Specify the dataset to evaluate.')
    
    parser.add_argument('--method', type=str, default='sdv', 
                        help='Specify the synthetic data generation method to evaluate')
    
    parser.add_argument('--privacy-percentile', type=float, default=None,
                        help='Parameter for privacy calculation')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def prepare_dataset(dataset, method):
    tables_synthetic = load_tables(dataset, data_type=f'synthetic/{method}')
    tables_original = load_tables(dataset, split='train')

    metadata = load_metadata(dataset)
    root_table = get_root_table(dataset)

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


def calculate_privacy(transformed_original, transformed_synthetic, **kwargs):
    if kwargs.get("privacy_percentile"):
        privacy_percentile = kwargs.get("privacy_percentile")
        return nearest_neighbor_distribution_score(transformed_original, transformed_synthetic, privacy_percetile=privacy_percentile)
    else:
        return nearest_neighbor_distribution_score(transformed_original, transformed_synthetic)

     
if __name__ == "__main__":

    args = get_args()

    transformed_original, transformed_synthetic = prepare_dataset(args.dataset, args.method)
    
    privacy_score = calculate_privacy(transformed_original, transformed_synthetic)

    print(f"Privacy score of {args.dataset}: {privacy_score}")