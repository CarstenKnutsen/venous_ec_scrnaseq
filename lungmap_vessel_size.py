
'''
Goal:Check lunmap BPD vessel_size_gradient
'''

import scanpy as sc
import scanpy.external as sce
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import palantir
import cellrank as cr
import scvelo as scv
from functions import palantir_psuedotime_routine_external_datasets,find_gene_overlap_in_pseudotimes,

adata_name='lungmap_bpd_2024'
figures = "data/figures/figures_tmp/lungmap_bpd_2024"
data = "data/single_cell_files/scanpy_files"

os.makedirs(figures, exist_ok=True)
sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5,1.5))
sc.settings.figdir = figures
sns.set_style('white', rc={
    'xtick.bottom': True,
    'ytick.left': True,
})
plt.rcParams["font.family"] = "Arial"
size=15
sc.settings.autoshow = False
if __name__ == '__main__':
    data = '/home/carsten/alvira_bioinformatics/data/external_datasets/lungmap_bpd_LMEX0000004400'
    human_adata = sc.read(f'{data}/BPD-adata_combined.h5ad')
    adata_obs_df = pd.read_csv(f'{data}/BPD_RNA_author-clusters.txt',sep='\t',header=0,index_col=0)
    for col in adata_obs_df:
        human_adata.obs[col] = adata_obs_df[col]
    sc.pp.calculate_qc_metrics(human_adata,inplace=True)
    sc.pp.normalize_total(human_adata,target_sum=1e4)
    sc.pp.log1p(human_adata)
    human_adata_endo =  human_adata[human_adata.obs['cell type'].isin(['AEC','CAP1','VEC'])]
    human_adata_endo.uns['Cell Subtype_colors']= ['#4A90E2','#9B59B6','#E35D6A']
    sc.pp.highly_variable_genes(human_adata_endo, batch_key='donor')
    sc.pp.pca(human_adata_endo)
    sce.pp.harmony_integrate(human_adata_endo, 'sample', adjusted_basis='X_pca_harmony',max_iter_harmony=20)
    sc.pp.neighbors(human_adata_endo, use_rep="X_pca_harmony")
    sc.tl.leiden(human_adata_endo, resolution=0.5)
    sc.tl.rank_genes_groups(human_adata_endo,'leiden',method='wilcoxon')
    sc.pl.rank_genes_groups_dotplot(human_adata_endo,dendrogram=False,save='endo_leiden.png')
    human_adata_endo=human_adata_endo[~human_adata_endo.obs['leiden'].isin(['5'])] # high mt
    human_adata_endo.obs['Cell Subtype'] = human_adata_endo.obs['leiden'].map({'0':'Cap1','1':'Cap1','2':'Arterial EC','3':'Venous EC','4':'Cap1'})
    sc.tl.umap(human_adata_endo,min_dist=0.5)
    start_and_end_states = {'root_ct': ('Cap1', 0, 'min'),
                            'terminal_cts': {'Arterial EC': (1, 'max'),
                                             'Venous EC': (0, 'max')
                                             }
                            }
    palantir_psuedotime_routine_external_datasets(human_adata_endo, start_and_end_states, save_prefix=figures,
                                                  adata_name=adata_name)
    find_gene_overlap_in_pseudotimes(human_adata_endo, save_prefix=figures, adata_name=adata_name)

