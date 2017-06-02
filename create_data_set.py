import numpy as np
import pandas as pd
from glob import glob
import os.path as osp
import os
from shutil import copy2


CODE_AUTISM = 1
CODE_CONTROL = 2


def ensure_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)


def get_df_pheno(pheno_path):
    return pd.read_csv(pheno_path)


def get_ids_from_pheno(df_pheno):
    # first row is somehow just an ABIDE_xx enumeration
    subj_ids = df_pheno.SubID.values[1:].astype(np.int)
    anon_ids = df_pheno['Unnamed: 0'].values[1:].astype(str)
    return anon_ids, subj_ids


def get_paper_ids(paper_path):
    return np.loadtxt(paper_path).astype(np.int)


def get_mprage(path_dir='/home/rthomas/abide/data'):
    """
    Assumes certain file structure. Won't work otherwise...
    :param path_dir:
    :return:
    """
    return np.array(sorted(glob(osp.join(path_dir, '*/*/*/*/*/*/*/*/*nii.gz'))))


def get_anon_ids_from_mprage(mprage_path):
    df_mprage = pd.DataFrame(data=mprage_path, columns=['strucs'])
    return df_mprage.strucs.str.extract('(A000\d{5})').values


def get_diagnosis_df(diag_path):
    return pd.read_csv(diag_path)


def take_paper_ids(anon_ids, subj_ids, paper_ids):
    df = pd.DataFrame(data=subj_ids, columns=['subj_ids'])
    idx_take = df.subj_ids.isin(paper_ids).values
    return anon_ids[idx_take.squeeze()], subj_ids[idx_take.squeeze()]


def store_meta_data(df_diag, subj_ids_to_take, save_path):
    df_diag_take = df_diag.loc[df_diag.SUB_ID.isin(subj_ids_to_take), :]
    df_diag_take.to_csv(osp.join(save_path, 'metadata_subj.csv'), index=False)


def run(**kwargs):
    paper_ids = get_paper_ids(kwargs['paper_path'])
    mprage_np = get_mprage(kwargs['path_dir'])
    mprage_anonym = get_anon_ids_from_mprage(mprage_np)
    df_pheno = get_df_pheno(kwargs['pheno_path'])
    df_diag = get_diagnosis_df(kwargs['diag_path'])
    anonym_ids, subj_ids = get_ids_from_pheno(df_pheno)
    anonym_ids_take, subj_ids_take = take_paper_ids(anonym_ids, subj_ids, paper_ids)

    save_folder = kwargs['save_folder']
    ensure_folder(save_folder)
    store_meta_data(df_diag, subj_ids_take, save_folder)

    ensure_folder(osp.join(save_folder, 'control'))
    ensure_folder(osp.join(save_folder, 'autism'))

    for i_mprage, struc_img in enumerate(mprage_np):
        print '{}/{}'.format(i_mprage + 1, mprage_np.size)

        subj_anonym = mprage_anonym[i_mprage]
        idx_subj = anonym_ids_take == subj_anonym

        if not np.any(idx_subj):
            print 'Skipping: {}'.format(subj_anonym)
            continue

        subj_id = subj_ids_take[idx_subj]
        subj_dx = df_diag.DX_GROUP[df_diag.SUB_ID == subj_id].values
        assert subj_dx.size == 1, 'Multiple subjects for id {}({})'.format(subj_id, subj_anonym)
        subj_dx = subj_dx[0]

        if subj_dx == CODE_CONTROL:
            new_file = osp.join(save_folder, 'control', 'con_{}_{}.nii.gz'.format(subj_id, subj_anonym))
        elif subj_dx == CODE_AUTISM:
            new_file = osp.join(save_folder, 'autism', 'aut_{}_{}.nii.gz'.format(subj_id, subj_anonym))
        else:
            raise RuntimeError('Subject {} ({}) has DX code {}.'.format(subj_id, subj_anonym, subj_dx))

        copy2(struc_img, new_file)


if __name__ == '__main__':
    main_path = '/home/rthomas/abide/data'

    kw = {'path_dir': main_path,
          'pheno_path': osp.join(main_path, 'abide_phenotype.csv'),
          'diag_path': osp.join(main_path, 'Phenotypic_V1_0b.csv'),
          'paper_path': osp.join(main_path, 'abide_ids.txt'),
          'save_folder': osp.join(main_path, 'deep_data')}
    run(**kw)
