import pandas as pd
import numpy as np

def fill_na(df):
    fill_na_strategy = {
        'sex': 'unk',
        'age': 0,
        'n_siblings_spouses': 0,
        'parch': 0,
        'fare': 0,
        'class': 'unk',
        'deck':'unk',
        'embark_town': 'unk',
        'alone': 'unk'
    }
    for col in fill_na_strategy:
        df.loc[df[col].isnull(), col] = fill_na_strategy[col]

def preprocess(df,cate_cols,existed_cate=None):
    if not existed_cate:
        categories = {}
        for c in cate_cols:
            categories[c] = df[c].unique().tolist()
            
            #make sure 'unk' has code 0
            if 'unk' in categories[c]:
                categories[c].remove('unk')
            categories[c]=['unk']+categories[c]

            df[c] = pd.Categorical(df[c],
                                     categories=categories[c]).codes
        return categories
    else:
        for c in cate_cols:
            df[c] = pd.Categorical(df[c],
                                     categories=existed_cate[c]).codes
        return
