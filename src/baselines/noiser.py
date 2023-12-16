import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import random

from sklearn import datasets

from src.data.utils import save_tables, load_tables, load_metadata, get_root_table, get_field_type

"""
params:
    par_cat - how many cat cols are noised
    par_num - how much numerical cols are noised
"""

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

def generate_noised_dataset(dataset_name, noise_fk=False):
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

        model = Noiser(tables_train[table], num_idx, cat_idx)

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

    save_tables(synthetic_data, dataset_name, data_type='synthetic/noiser')


if __name__ == '__main__':
    # iris = datasets.load_iris(as_frame=True, return_X_y=True)
    # iris = pd.concat((iris[1], iris[0]), axis=1)

    # model = Noiser(iris, [0, 1, 2, 3], [4], noise_level=0.1, permut_probability=0.5)
    # noised_iris = model.fit_transform()
    # print(iris)
    # print(noised_iris)

    generate_noised_dataset('rossmann-store-sales')
    generate_noised_dataset('mutagenesis')