"""
Goal: Run cell communication using liana
Author:Carsten Knutsen
Date:250318
conda_env:liana
"""


import scanpy as sc
import scanpy.external as sce
import pandas as pd
import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functions import plot_obs_abundance,compare_obs_values_within_groups_to_excel
import liana as li
from gprofiler import GProfiler


adata_name = "venous_ec"
output = "data/figures/ccc"
os.makedirs(output, exist_ok=True)
data = "data/single_cell_files/scanpy_files"
os.makedirs(output, exist_ok=True)
sc.set_figure_params(dpi=300, dpi_save=300, format="png")
sc.settings.figdir = output
sc.settings.autoshow = False


def convert_mouse_to_human(adata):
    """Convert var_names in an AnnData object from mouse to human gene names using gProfiler."""
    gp = GProfiler(return_dataframe=True)

    # Convert mouse genes to human
    conversion_df = gp.orth(organism='mmusculus',query=adata.var_names.tolist(), target='hsapiens')

    # Extract mapping
    mouse_to_human = dict(zip(conversion_df['incoming'], conversion_df['name']))
    # Map `var_names`
    adata.var_names = adata.var_names.map(lambda x: mouse_to_human.get(x) if mouse_to_human.get(x) not in [None, 'N/A'] else x)
    adata.var_names = adata.var_names.str.upper() #Capitalize any unmapped genes
    # Remove duplicates (if any)
    adata.var_names_make_unique()
    return adata

if __name__ == "__main__":
    adata = sc.read(f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad")
    convert_mouse_to_human(adata)
    li.mt.rank_aggregate(adata,
                         groupby='Cell Subtype_no_cc',
                         resource_name='consensus',
                         expr_prop=0.1,
                         use_raw=False,
                         verbose=True)
    df = adata.uns['liana_res']
    df = df.sort_values('spec_weight', ascending=False)
    df.to_csv(f'{output}/ccc_all_cells.csv')
    vessel_adata = sc.read(f'{data}/{adata_name}_vessel_size_adata.gz.h5ad',compression='gzip')
    convert_mouse_to_human(vessel_adata)
    li.mt.rank_aggregate(vessel_adata,
                         groupby='Cell Subtype_size',
                         resource_name='consensus',
                         expr_prop=0.1,
                         use_raw=False,
                         verbose=True)
    df = vessel_adata.uns['liana_res']
    df = df.sort_values('spec_weight', ascending=False)
    df.to_csv(f'{output}/ccc_vessel_size.csv')