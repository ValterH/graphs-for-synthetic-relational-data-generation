import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

import json

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

from src.data.utils import load_tables, load_metadata

import argparse

"""
How to call:

python src/eval/eval_density.py --dataname [NAME_OF_DATASET] --model tabsyn --path [PATH_TO_SYNTHETIC_DATA]

"""

parser = argparse.ArgumentParser()
# dataset to evaluate
parser.add_argument('--dataset', type=str, default='rossmann-store-sales',
                    help='Specify the dataset to evaluate.')

parser.add_argument('--method', type=str, default='sdv', 
                    help='Specify the synthetic data generation method to evaluate')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()

if __name__ == '__main__':

    tables_synthetic = load_tables(args.dataset, data_type=f'synthetic/{args.method}')
    tables_original = load_tables(args.dataset, split='train')


    tables_synthetic = load_tables(args.dataset, data_type=f'synthetic/{args.method}')
    tables_original = load_tables(args.dataset, split='train')

    metadata = load_metadata(args.dataset)

    for table in tables_original.keys():

        save_dir = f'eval/density/{args.method}/{args.dataset}/{table}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        table_metadata = metadata.to_dict()['tables'][table]
        syn_data = tables_synthetic[table]
        real_data = tables_original[table]

        for column in real_data.columns:
            synth_col = syn_data[column]
            orig_col = real_data[column] 
            if table_metadata['fields'][column]['type'] == 'categorical':
                synth_col = synth_col.fillna('nan')
                orig_col = orig_col.fillna('nan')
            elif table_metadata['fields'][column]['type'] == 'numerical':
                synth_col = synth_col.fillna(0)
                orig_col = orig_col.fillna(0)
            elif table_metadata['fields'][column]['type'] == 'id':
                continue
            plt.clf()
            _, bins, _ = plt.hist(orig_col, density=True, stacked=True, bins='fd',label='Original')
            plt.hist(synth_col, alpha=0.5, bins=bins, density=True, stacked=True, label='Synthetic')
            plt.legend()
            plt.savefig(f'{save_dir}/{column}.png')

        

        qual_report = QualityReport()
        qual_report.generate(real_data, syn_data, table_metadata)

        diag_report = DiagnosticReport()
        diag_report.generate(real_data, syn_data, table_metadata)

        quality =  qual_report.get_properties()
        diag = diag_report.get_properties()

        Shape = quality['Score'][0]
        Trend = quality['Score'][1]

        with open(f'{save_dir}/quality.txt', 'w') as f:
            f.write(f'{Shape}\n')
            f.write(f'{Trend}\n')

        Quality = (Shape + Trend) / 2

        shapes = qual_report.get_details(property_name='Column Shapes')
        trends = qual_report.get_details(property_name='Column Pair Trends')
        coverages = diag_report.get_details('Coverage')


        shapes.to_csv(f'{save_dir}/shape.csv')
        trends.to_csv(f'{save_dir}/trend.csv')
        coverages.to_csv(f'{save_dir}/coverage.csv')
