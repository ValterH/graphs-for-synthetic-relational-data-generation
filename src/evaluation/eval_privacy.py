import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


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
def nearest_neighbor_distribution_score(original, synthetic, **kwargs):
    """
    Calculate graphVAE's privacy score
    """
    # split original dataset in half for alpha calculation 
    D_1, D_2 = train_test_split(original, test_size=0.5)

    # calculate the 5th/x-th percentile 
    ratios = calculate_ratios(D_1, D_2, **kwargs)
    alpha = np.percentile(ratios, 5)

    # calculate the ratios with synthetic data
    ratios_syn = calculate_ratios(original, synthetic, **kwargs)

    privacy = np.mean(ratios_syn < alpha)

    return privacy


def calculate_ratios(D_1, D_2, **kwargs):
    if kwargs["privacy_percetile"]:
        privacy_percentile = kwargs["privacy_percetile"]
    else:
        privacy_percentile = 0.05
    
    # calculate nearest neighbor distances for D_1 with itself and D_1 with D_2
    self_distance = pairwise_distances(D_1, matric="manhattan", **kwargs)
    self_distance = np.min(self_distance, axis=1)
    mutual_distance = pairwise_distances(D_1, D_2, matric="manhattan", **kwargs)
    mutual_distance = np.min(mutual_distance, axis=1)

    ratios = mutual_distance / self_distance
    
    return ratios
     
if __name__ == "__main__":
    # df1 = pd
    pass