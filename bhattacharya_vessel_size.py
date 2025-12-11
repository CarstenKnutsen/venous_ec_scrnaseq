'''
Goal:Check bhattacharya2024 neonatal lung for vessel_size_gradient
conda_env:vessel_size
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
from functions import palantir_psuedotime_routine_external_datasets,find_gene_overlap_in_pseudotimes
figures = "data/figures/figures_tmp/bhattacharya_2024"
data_fol = '/home/carsten/alvira_bioinformatics/data/external_datasets/'
adata_name = 'bhattacharya_2024'

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
    human_adata = sc.read(f'{data_fol}/bhattacharya_2024.h5ad')
    human_adata.var_names = human_adata.var['feature_name']
    del human_adata.raw
    sc.pp.highly_variable_genes(human_adata, batch_key='Sample')
    sc.pp.pca(human_adata)
    sce.pp.harmony_integrate(human_adata, 'Sample', adjusted_basis='X_pca')
    sc.pp.neighbors(human_adata)
    sc.tl.leiden(human_adata)
    sc.tl.umap(human_adata)
    sc.pl.umap(human_adata, color='leiden', use_raw=False, save=f'{adata_name}_leiden.png',show=False)
    sc.pl.umap(human_adata, color='cell_type', use_raw=False, save=f'{adata_name}_old_celltype.png',show=False)
    sc.pl.umap(human_adata, color='leiden', use_raw=False, save=f'{adata_name}_leiden.png',show=False)
    leiden_dict = {'0': 'ASM', '1': 'AlF', '2': 'VSM', '3': 'Cap1', '4': 'Per', '5': 'Per', '6': 'Cap2', '7': 'AlM',
                   '8': 'AdF', '9': 'Venous EC', '10': 'MyF', '11': 'AEC', '12': 'Nkc',
                   '13': 'Arterial EC', '14': 'Tce', '15': 'Bce', '16': 'Lym', '17': 'InM', '18': 'AdF', '19': 'Cil'}
    human_adata.obs['Cell Subtype'] = [leiden_dict[x] for x in human_adata.obs['leiden']]
    sc.pl.umap(human_adata, color='Cell Subtype', save=f'{adata_name}_new_celltype.png',show=False)
    cts = ['Arterial EC','Cap1','Venous EC']
    human_adata = human_adata[human_adata.obs['Cell Subtype'].isin(cts)]
    sc.pp.highly_variable_genes(human_adata,batch_key='Sample')
    sc.pp.pca(human_adata, mask_var="highly_variable")
    sce.pp.harmony_integrate(human_adata, 'Sample', adjusted_basis='X_pca_harmony',max_iter_harmony=20)
    sc.pp.neighbors(human_adata, use_rep="X_pca_harmony")
    sc.tl.leiden(human_adata, resolution=0.5)
    human_adata.obs['Cell Subtype'] = pd.Categorical(human_adata.obs['leiden'].map({'0':'Cap1','1':'Venous EC','2':'Cap1','3':'Arterial EC','4':'Cap1'}),categories = cts)
    human_adata.uns['Cell Subtype_colors']= ['#4A90E2','#9B59B6','#E35D6A']

    sc.tl.umap(human_adata, min_dist=2.5)
    sc.pl.umap(human_adata, color='Cell Subtype', save=f'{adata_name}_endos.png',show=False)


    start_and_end_states = {'root_ct':('Cap1',1,'max'),
                            'terminal_cts' :{'Arterial EC':(1,'max'),
                                              'Venous EC':(1,'min')
                                              }
                            }
    palantir_psuedotime_routine_external_datasets(human_adata,start_and_end_states,save_prefix=figures,adata_name=adata_name)
    find_gene_overlap_in_pseudotimes(human_adata,save_prefix=figures,adata_name=adata_name)














