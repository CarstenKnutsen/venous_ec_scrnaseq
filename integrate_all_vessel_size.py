
'''
Goal:Integrate all vessel size datasets together
'''
import scanpy as sc
import scanpy.external as sce
import os 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import palantir
import gprofiler
import anndata
from gprofiler import GProfiler

figures = "data/figures/figures_tmp/"
figures_out = f'{figures}/integrate_all'
os.makedirs(figures_out, exist_ok=True)
sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5,1.5))
sc.settings.figdir = figures_out
sns.set_style('white', rc={
    'xtick.bottom': True,
    'ytick.left': True,
})
plt.rcParams["font.family"] = "Arial"
size=5
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


if __name__ == '__main__':
    adata_dict = {}

    adata = sc.read(f'{figures}/hurskainen_2021/vessel_size.gz.h5ad')
    adata.x = adata.layers['log1p'].copy()
    del adata.layers
    adata = convert_mouse_to_human(adata)
    adata.obs['Species'] = 'Mouse'
    adata.obs['Age'] = 'Neonatal'
    adata_dict['Hurskainen 2021'] = adata

    adata = sc.read(f'{figures}/tabula_muris_senis/vessel_size.gz.h5ad')
    del adata.layers
    print(adata.var_names)
    adata = convert_mouse_to_human(adata)
    print(adata.var_names)
    del adata.raw
    adata.obs['Species'] = 'Mouse'
    adata.obs['Age'] = 'Adult'
    adata_dict['Tabula Muris Senis 2020'] = adata

    adata = sc.read(f"data/single_cell_files/scanpy_files/venous_ec_vessel_size_plot.gz.h5ad")
    adata.X = adata.layers['log1p']
    del adata.layers
    adata = convert_mouse_to_human(adata)
    adata.obs['Cell Subtype'] = adata.obs['Cell Subtype_no_cc'].copy()
    adata.obs['Species'] = 'Mouse'
    adata.obs['Age'] = 'Neonatal'
    adata_dict['Sveiven 2025'] = adata

    adata = sc.read(f'{figures}/bhattacharya_2024/vessel_size.gz.h5ad')
    del adata.layers
    adata.obs['Species'] = 'Human'
    adata.obs['Age'] = 'Neonatal'
    adata_dict['Bhattacharya 2024'] = adata

    adata = sc.read(f'{figures}/tabula_sapiens/vessel_size.gz.h5ad')
    del adata.layers
    adata.obs['Species'] = 'Human'
    adata.obs['Age'] = 'Adult'
    adata_dict['Tabula Sapiens v2 2025'] = adata
    adata = sc.read(f'{figures}/lungmap_bpd_2024/vessel_size.gz.h5ad')
    del adata.layers
    adata_dict['LungMAP 2024'] = adata
    adata.obs['Species'] = 'Human'
    adata.obs['Age'] = 'Neonatal'
    for x in adata_dict:
        print(x)
        print(adata_dict[x].var_names)

    adata = anndata.concat(adata_dict.values(), join='inner', label='Dataset', keys=adata_dict.keys(), index_unique=None)
    del adata.obsm
    adata.obs_names_make_unique()
    nindex = np.random.permutation(adata.obs.index)
    adata=adata[nindex,:]
    ## read in cell cycle genes
    cell_cycle_genes = [x.strip() for x in open('/home/carsten/alvira_bioinformatics/venous_ec_scrnaseq/data/outside_data/regev_lab_cell_cycle_genes.txt')]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    adata.layers['log1p'] = adata.X.copy()
    sc.pp.regress_out(adata, ['S_score', 'G2M_score'])
    adata.uns['Cell Subtype_colors']= ['#4A90E2','#9B59B6','#E35D6A']
    sc.pp.highly_variable_genes(adata, batch_key='Dataset',n_top_genes=200)
    sc.pp.pca(adata,use_highly_variable=True)
    sce.pp.bbknn(adata,batch_key='Dataset',neighbors_within_batch=2,set_op_mix_ratio=5)
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.umap(adata,min_dist=0.5)
    adata.X = adata.layers['log1p'].copy()
    sc.tl.rank_genes_groups(adata,'leiden',method='wilcoxon')
    sc.pl.rank_genes_groups_dotplot(adata,dendrogram=False)
    sc.pl.umap(adata,color=['leiden'],cmap='Oranges',size=size,frameon=False,save='_all_leiden.png')
    sc.pl.umap(adata,color=['Vessel size score'],cmap='Oranges',size=size,frameon=False,save='_all_vessel_size_score.png')
    sc.pl.umap(adata,color=['Vessel size category'],cmap='viridis',size=size,frameon=False,save='_allvessel_size_category.png')
    sc.pl.umap(adata,color=['Cell Subtype'],cmap='viridis',size=size,legend_fontsize=10, legend_fontoutline=1,frameon=False,save='_all_cellsubtype.png')
    sc.pl.umap(adata,color=['MGP'],cmap='viridis',frameon=False,size=size,save='_all_mgp.png')
    sc.pl.umap(adata,color=['COL4A1'],cmap='viridis',frameon=False,size=size,save='_all_col4a1.png')
    sc.pl.umap(adata,color=['COL4A2'],cmap='viridis',frameon=False,size=size,save='_all_col4a2.png')
    sc.pl.umap(adata,color=['ELN'],cmap='viridis',frameon=False,size=size,save='_all_eln.png')
    sc.pl.umap(adata,color=['FBLN2'],cmap='viridis',frameon=False,size=size,save='_all_fbln2.png')
    sc.pl.umap(adata,color=['HEY1'],cmap='viridis',frameon=False,size=size,save='_all_hey1.png')
    sc.pl.umap(adata,color=['NR2F2'],cmap='viridis',frameon=False,size=size,save='_all_nr2f2.png')
    sc.pl.umap(adata,color=['Dataset'],cmap='viridis',frameon=False,size=size,save='_all_dataset.png')
    sc.pl.umap(adata,color=['MKI67'],cmap='viridis',frameon=False,size=size,save='_all_mki67.png')
    sc.pl.umap(adata,color=['Age'],palette=['purple','yellow'],frameon=False,size=size,save='_all_age.png')
    sc.pl.umap(adata,color=['Species'],cmap='viridis',frameon=False,size=size,save='_all_species.png')
    sc.pl.umap(adata,color=['Dataset'],cmap='viridis',frameon=False,size=size,save='_all_dataset.png')
    sc.pl.umap(adata,color='Vessel size category')
    for dataset in adata.obs['Dataset'].unique():
        sc.pl.umap(adata.copy(),color=['Dataset'],cmap='viridis',mask_obs=adata.obs['Dataset']==dataset,title='',legend_loc=None,frameon=False,size=size,save=F'_{dataset}_dataset.png')
        sc.pl.umap(adata.copy(),color='Vessel size category',mask_obs=adata.obs['Dataset']==dataset,title='',legend_loc=None,frameon=False,size=size,save=F'_{dataset}_dataset_vessel_size.png')
        sc.pl.umap(adata.copy(),color='Vessel size score',mask_obs=adata.obs['Dataset']==dataset,cmap='Oranges',title='',colorbar_loc=None,frameon=False,size=size,save=F'_{dataset}_dataset_vessel_size_score.png')

