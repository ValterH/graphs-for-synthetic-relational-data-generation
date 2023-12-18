import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import random
import argparse

from sklearn import datasets

from src.data.utils import save_tables, load_tables, load_metadata, get_root_table, get_field_type, merge_children
from src.evaluation.eval_privacy import get_args, calculate_privacy, drop_ids
from src.evaluation.eval_classifier import CustomHyperTransformer


"""
params:
    par_cat - how many cat cols are noised
    par_num - how much numerical cols are noised
"""

NOISE_SPACE = np.logspace(-3, 1, 100)


class Noiser:
    def __init__(self, dataset, num_index, cat_index, noise_level=0.1, permut_probability=0.1, return_original=False):
        """
        Initialize the Noiser object.

        Parameters:
        - dataset: pandas dataframe.
        - noise_level: The level of noise to be added to the dataset.
        - return_original: whether to return the original dataset (default=False).
        """
        self.dataset = dataset
        self.num_index = num_index
        self.cat_index = cat_index
        self.noise_level = noise_level
        self.permut_probability = permut_probability
        self.return_original = return_original
        self.noised_dataset = None

    def fit_transform(self):
        df_num = self.dataset.iloc[:, self.num_index]
        df_cat = self.dataset.iloc[:, self.cat_index]

        df_num = self.transform_num(df_num)
        df_cat = self.transorm_cat(df_cat)

        df = self.dataset.copy()
        for i in range(self.dataset.shape[1]):
            if i in self.num_index:
                col = df_num.pop(df_num.columns[0])
                df[col.name] = col.values
            else:
                col = df_cat.pop(df_cat.columns[0])
                df[col.name] = col.values

        return df

    def transform_num(self, df):
        for col in df.columns:
            # check if integer in reality
            if np.mean(df[col] % 1) < 0.0000001:
                integer = True
            else:
                integer = False
            
            var = np.var(df[col])
            df[col] = df[col].apply(lambda x : np.random.normal(x, np.sqrt(var)*self.noise_level))
            if integer:
                df[col] = df[col].apply(lambda x: round(x))
        return df

    def transorm_cat(self, df):
        for col in df.columns:
            # calculate probabilities of each category in the column
            value_counts = df[col].value_counts(normalize=True)
            # change values with shuffle()
            df[col] = df[col].apply(lambda x : self.shuffle(x, value_counts.index, value_counts))
        return df

    def shuffle(self, orig, vals, p_vals):
        # change value with probability of self.permut_probability
        if bernoulli(1 - self.permut_probability).rvs(1) == 1:
            return orig
        else:
            # if we do change it we draw from possible values w.r.t. their probabilities
            return random.choices(vals, p_vals)[0]


class MissingColumnError(Exception):
    pass

def generate_noised_dataset(dataset_name, noise_fk=False, save_=True, noise_level_num=None, noise_level_cat=None, merge=False):
    tables_train = load_tables(dataset_name, split='train')
    metadata = load_metadata(dataset_name)

    synthetic_data = {}
    for table in tables_train:
        # remove id column/-s
        pr_key = metadata.get_primary_key(table)
        id_col_idx = tables_train[table].columns.get_loc(pr_key)
        id_col = tables_train[table].pop(pr_key)

        # remove foreign keys also if necessary
        if not noise_fk:
            fk_col_idxs = []
            fk_cols = []
            for i, col in enumerate(tables_train[table].columns):
                type_ = get_field_type(table, col, metadata._metadata)
                if type_ == "id":
                    fk_col_idxs.append(i)
                    fk_cols.append(tables_train[table].pop(col))

        # create index lists for cat and num cols
        cat_idx = []
        num_idx = []
        for i, col in enumerate(tables_train[table].columns):
            type_ = get_field_type(table, col, metadata._metadata)
            if type_ == "categorical":
                cat_idx.append(i)
            elif type_ == "id": # only foreign keys at this point
                cat_idx.append(i)
            elif type_ == "numerical":
                num_idx.append(i)
            elif type_ == "datetime":
                num_idx.append(i)
                dff = tables_train[table]
                dff[col] = dff[col].apply(lambda x : pd.to_datetime(x).timestamp())
                tables_train[table] = dff
            else:
                raise MissingColumnError("Column in dataset not in metadata")    

        if noise_level_num:
            if noise_level_cat:
                model = Noiser(tables_train[table], 
                               num_idx, 
                               cat_idx, 
                               noise_level=noise_level_num, 
                               permut_probability=noise_level_cat)
            else:
                model = Noiser(tables_train[table], 
                               num_idx, 
                               cat_idx, 
                               noise_level=noise_level_num)
        elif noise_level_cat:
            model = Noiser(tables_train[table], 
                            num_idx, 
                            cat_idx, 
                            permut_probability=noise_level_cat)
        
        else:
            model = Noiser(tables_train[table], 
                            num_idx, 
                            cat_idx)      
        # model_save_path = f'models/noiser/{dataset_name}/model.pickle'
        # if not os.path.exists(os.path.dirname(model_save_path)):
        #     os.makedirs(os.path.dirname(model_save_path))
        # with open(model_save_path, "wb") as f:
        #     pickle.dump(model, open(model_save_path, "wb" ) )

        synthetic_table = model.fit_transform()

        # change back the date columns
        for col in tables_train[table].columns:
            if get_field_type(table, col, metadata._metadata) == "datetime":
                synthetic_table[col] = synthetic_table[col].apply(lambda x : pd.to_datetime(x, unit="s"))
        
        # add back foreign keys if necessary
        if not noise_fk:
            for loc in fk_col_idxs:
                col = fk_cols.pop(0)
                synthetic_table.insert(loc=loc, column=col.name, value=col)
                
        # add back id column
        synthetic_table.insert(loc=id_col_idx, column=id_col.name, value=id_col)

        synthetic_data[table] = synthetic_table

    if merge:
        root_table = get_root_table(dataset_name)

        synthetic = merge_children(synthetic_data, metadata, root_table)

        # drop all foreign and primary keys
        for table in metadata.to_dict()['tables'].keys():
            drop_ids(synthetic, table, metadata)

        transformed_synthetic = synthetic.copy()
        
        column_names = transformed_synthetic.columns.to_list()
        transformed_synthetic = transformed_synthetic.reindex(column_names, axis=1)

        max_items = 100000

        if 'Date' in column_names:
            transformed_synthetic['Date'] = pd.to_numeric(pd.to_datetime(transformed_synthetic['Date']))

        ht = CustomHyperTransformer()
        transformed_synthetic = ht.fit_transform(transformed_synthetic)
        
        transformed_synthetic.to_csv(f"data/synthetic/noiser/{dataset_name}/merged/noise_{noise_level_num}.csv")

        return transformed_synthetic

    if save_:
        save_tables(synthetic_data, dataset_name, data_type='synthetic/noiser')
    else:
        return synthetic_data

def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Noise parameters')

    # dataset to evaluate
    parser.add_argument('--dataset', type=str, default='rossmann-store-sales',
                        help='Specify the dataset to evaluate.')
    
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Parameter for noise levels introduced')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    
    params = {
        "save_" : True,
        "merge" : False
    }
    generate_noised_dataset(args.dataset, 
                            noise_level_num=args.noise_level, 
                            noise_level_cat=args.noise_level, 
                            save_=params["save_"],
                            merge=params["merge"])
    