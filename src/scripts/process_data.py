import pandas as pd

# Rossmann Store Sales Dataset
df = pd.read_csv('data/rossmann-store-sales/original/store.csv')
df_ordered = pd.read_csv('tabsyn/data/store/ordered.csv')
column_order = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'StoreType', 'Assortment', 'Promo2', 'PromoInterval', 'Store']
df = df.reindex(columns=column_order)

num_columns = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

# replace Nans with 0 constants
for col in num_columns:
    df[col] = df[col].fillna(0)

df.to_csv('data/rossmann-store-sales/store.csv', index=False)

df = pd.read_csv('data/rossmann-store-sales/original/test.csv')
# Date,DayOfWeek,Open,Promo,StateHoliday,SchoolHoliday,Store,Id
column_order = ['Date', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Store', 'Id']
df = df.reindex(columns=column_order)
df.to_csv('data/rossmann-store-sales/test.csv', index=False)