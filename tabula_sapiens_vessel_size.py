
'''
Goal:Check tabula sapiens lung for vessel_size_gradient
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

adata_name='tabula_sapiens'
figures = "data/figures/figures_tmp/tabula_sapiens"
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
    adata_ts = sc.read('/home/carsten/alvira_bioinformatics/data/external_datasets/tabula_sapiens/Lung_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad')
    adata_ts.X = adata_ts.layers['log_normalized'].copy()
    adata_ts.obs['free_annotation'].cat.categories
    vessel_cts = ['arterial endothelial cell','capillary endothelial cell','vein endothelial cell']
    adata_ts_vessel = adata_ts[adata_ts.obs['free_annotation'].isin(vessel_cts)]
    sc.pp.highly_variable_genes(adata_ts_vessel)
    sc.pp.pca(adata_ts_vessel, mask_var="highly_variable")
    sce.pp.harmony_integrate(adata_ts_vessel,'donor',max_iter_harmony = 20)
    sc.pp.neighbors(adata_ts_vessel, use_rep="X_pca_harmony")
    sc.tl.leiden(adata_ts_vessel, resolution=0.5)
    sc.tl.rank_genes_groups(adata_ts_vessel,'leiden',method='wilcoxon')
    sc.tl.umap(adata_ts_vessel, min_dist=1)
    sc.pl.rank_genes_groups_dotplot(adata_ts_vessel,n_genes=10,save='endos_leiden.png')
    adata_ts_vessel_sub = adata_ts_vessel[~adata_ts_vessel.obs['leiden'].isin(['0','2','5','7','8'])] # Mix of donor specific celltypes, systemic VEC, low-quality
    sc.pp.highly_variable_genes(adata_ts_vessel_sub)
    sc.pp.pca(adata_ts_vessel_sub, mask_var="highly_variable")
    sce.pp.harmony_integrate(adata_ts_vessel_sub,'donor',max_iter_harmony = 20)
    sc.pp.neighbors(adata_ts_vessel_sub, use_rep="X_pca_harmony")
    sc.tl.leiden(adata_ts_vessel_sub, resolution=0.2)
    leiden_ct_dict = {'0':'Cap1','1':'Venous EC','2':'Arterial EC'}
    adata_ts_vessel_sub.obs['Cell Subtype'] = [leiden_ct_dict[x] for x in adata_ts_vessel_sub.obs['leiden']]
    adata_ts_vessel_sub.uns['Cell Subtype_colors']= ['#4A90E2','#9B59B6','#E35D6A']
    sc.tl.umap(adata_ts_vessel_sub, min_dist=1)
    start_and_end_states = {'root_ct': ('Cap1', 0, 'max'),
                            'terminal_cts': {'Arterial EC': (1, 'min'),
                                             'Venous EC': (0, 'min')
                                             }
                            }
    palantir_psuedotime_routine_external_datasets(adata_ts_vessel_sub, start_and_end_states, save_prefix=figures,
                                                  adata_name=adata_name)
    find_gene_overlap_in_pseudotimes(adata_ts_vessel_sub, save_prefix=figures, adata_name=adata_name)


