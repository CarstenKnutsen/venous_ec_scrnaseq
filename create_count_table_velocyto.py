'''Goal:Create count table for all cells after SoupX cleanup of tbx2 ko samples
Date:231222
Author: Carsten Knutsen
conda_env:trajectory_inference
'''


import os
import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
import string
import anndata
from gtfparse import read_gtf
from anndata import AnnData
from collections import defaultdict
data = 'data/single_cell_files/velocyto'
output = 'data/single_cell_files/scanpy_files'
adata_name = "venous_ec"

os.makedirs(output, exist_ok=True)
if __name__ == '__main__':
    runs = os.listdir(data)
    adatas = []
    for run in runs:
        print(run)
        fn = f'{data}/{run}'
        adata_v = sc.read(fn)
        adata_v.var_names_make_unique()
        adata_v.obs_names = adata_v.obs_names.str.replace(':', '_')  # to match original data
        adata_v.obs_names = adata_v.obs_names.str.rstrip('x')
        adata_v.obs_names = [x + '-1' for x in adata_v.obs_names]
        sort, treat, time = run.split("_")
        adata_v.obs["Library"] = run
        adata_v.obs["Treatment"] = treat[0]
        adata_v.obs["Timepoint"] = time.split('.')[0]
        adata_v.obs["Sort"] = sort
        adatas.append(adata_v.copy())
    adata_v = anndata.concat(adatas)
    adata_v.obs["Treatment"].replace({"H": "Hyperoxia", "N": "Normoxia"}, inplace=True)
    print(adata_v)
    adata_v.write(f'{output}/{adata_name}_all_cells_velocyto.gz.h5ad', compression='gzip')
