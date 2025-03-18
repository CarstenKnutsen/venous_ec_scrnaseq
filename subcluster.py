"""
Goal: Subcluster each cell type to see diversity
Author:Carsten Knutsen
Date:250103
conda_env:venous_ec
"""


import scanpy as sc
import scanpy.external as sce
import pandas as pd
import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib.pyplot as plt
import os
from functions import plot_obs_abundance,compare_obs_values_within_groups_to_excel


adata_name = "venous_ec"
output = "data/figures"
os.makedirs(output, exist_ok=True)
data = "data/single_cell_files/scanpy_files"
os.makedirs(output, exist_ok=True)
sc.set_figure_params(dpi=300, dpi_save=300, format="png")
sc.settings.figdir = output
def subcluster_celltype(adata,obs,celltype,output_fol):
    output_ct = f'{output_fol}/{celltype}'
    os.makedirs(output_ct, exist_ok=True)
    sc.set_figure_params(dpi=300, format="png")
    sc.settings.figdir = output_ct
    ct_adata = adata[adata.obs[obs]==celltype]
    try:
        sc.pp.highly_variable_genes(ct_adata,
                                    batch_key="Library"
                                    )
    except:
        sc.pp.highly_variable_genes(ct_adata
                            )
    sc.pp.pca(ct_adata, use_highly_variable=True)
    sc.pl.pca_variance_ratio(ct_adata, show=False, save=True)
    sc.pl.pca_loadings(
        ct_adata,
        components=",".join([str(x) for x in range(1, 10)]),
        show=False,
        save=True,
    )
    sc.pp.neighbors(ct_adata, use_rep='X_pca')
    sc.tl.leiden(
        ct_adata,
        key_added=f"leiden_{celltype}",
        resolution=0.5
    )
    sc.tl.umap(ct_adata,min_dist=0.5)
    sc.tl.rank_genes_groups(ct_adata, f"leiden_{celltype}",pts=True, method="wilcoxon")
    print(ct_adata.obs[f"leiden_{celltype}"].cat.categories)
    sc.pl.rank_genes_groups_dotplot(
        ct_adata,
        groupby=f"leiden_{celltype}",
        n_genes=int(100 / len(ct_adata.obs[f"leiden_{celltype}"].unique())),
        show=False,
        save=f"{celltype}_leiden_markers.png",
    )
    lineage = ct_adata.obs['Lineage'].values[0]
    for color in [f'leiden_{celltype}','Library','Treatment','Cell Subtype','doublet_score',f'leiden_{lineage}','proliferation_score','phase','log1p_n_genes_by_umis','log1p_total_umis']:
        sc.pl.umap(ct_adata, color = color, show=False,save=color)
    with pd.ExcelWriter(
        f"{output_ct}/{celltype}_leiden_markers.xlsx", engine="xlsxwriter") as writer:
        for ld in sorted(ct_adata.obs[f"leiden_{celltype}"].unique()):
            df = sc.get.rank_genes_groups_df(
                ct_adata, key="rank_genes_groups", group=ld
            )
            df.to_excel(writer, sheet_name=f"{ld} v rest"[:31])
    ct_adata.write(f'{output_ct}/{celltype}_adata.gz.h5ad', compression='gzip')
    plot_obs_abundance(ct_adata,f"leiden_{celltype}",hue="Treatment",ordered=True,as_percentage=True,save=f'{output_ct}/{celltype}_leiden.png',hue_order=['Normoxia','Hyperoxia'])
    try:
     compare_obs_values_within_groups_to_excel(ct_adata,f"leiden_{celltype}",
                                             output_prefix=f"{output_ct}/leiden_{celltype}_comparisons")
    except:
        print(celltype)
if __name__ == "__main__":
    adata = sc.read(f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad")
    adata.uns['log1p']['base'] = None
    adata = adata[:,(adata.var['mt']==False)&(adata.var['ribo']==False)&(adata.var['hb']==False)]
## Subcluster each cell type
    for celltype in adata.obs['Cell Subtype'].unique():
        subcluster_celltype(adata,'Cell Subtype',celltype,f'{output}/subcluster')
    for celltype in adata.obs['Cell Subtype_no_cc'].unique():
        subcluster_celltype(adata,'Cell Subtype_no_cc',celltype,f'{output}/subcluster_no_cc')