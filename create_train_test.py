from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from create_data_set import CODE_AUTISM, CODE_CONTROL


def get_diag_df(diag_path):
    return pd.read_csv(diag_path)


def run(**kwargs):
    df_diag = get_diag_df(kwargs['diag_path'])

    code = pd.get_dummies(df_diag.SITE_ID).values
    code = code.dot(np.arange(1, code.shape[1] + 1))
    subj_ids = df_diag.SUB_ID.values

    subj_train, subj_test, _, code_test = train_test_split(subj_ids, code, test_size=200, stratify=code)
    subj_valid, subj_test = train_test_split(subj_test, test_size=100, stratify=code_test)

    df_diag['train'] = np.zeros(df_diag.shape[0], dtype=np.bool)
    df_diag['valid'] = np.zeros(df_diag.shape[0], dtype=np.bool)
    df_diag['test'] = np.zeros(df_diag.shape[0], dtype=np.bool)

    df_diag.loc[df_diag.SUB_ID.isin(subj_train), 'train'] = True
    df_diag.loc[df_diag.SUB_ID.isin(subj_valid), 'valid'] = True
    df_diag.loc[df_diag.SUB_ID.isin(subj_test), 'test'] = True

    df_diag['label'] = np.zeros(df_diag.shape[0], dtype=np.int)
    df_diag.loc[df_diag.DX_GROUP == CODE_CONTROL, 'label'] = 0
    df_diag.loc[df_diag.DX_GROUP == CODE_AUTISM, 'label'] = 1

    df_diag.to_csv(kwargs['diag_path'], index=False)


if __name__ == '__main__':
    kw = {'diag_path': '/home/rthomas/abide/data/deep_data/metadata_subj.csv'}
    run(**kw)
