
'''
Goal:Check hurskainen vessel_size_gradient
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
from functions import palantir_psuedotime_routine_external_datasets,find_gene_overlap_in_pseudotimes,custom_dotplot

adata_name='hurskainen_2021'
figures = "data/figures/figures_tmp/hurskainen_2021"
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


    source = '/home/carsten/alvira_bioinformatics/data/external_datasets/hurskainen2021_hyperoxia_lung/raw'
    metadata = pd.read_csv(os.path.join(source, "GSE151974_cell_metadata_postfilter.csv.gz"),
                           compression='gzip',
                           header=0,
                           index_col= 0,
                           sep=',',
                          )

    adata = sc.read_csv(os.path.join(source,"GSE151974_raw_umi_matrix_postfilter.csv.gz"),
                           first_column_names=True,
                           delimiter=',',
                        ).T
    adata.obs = metadata
    adata.obs.rename(columns = {'Age':'Timepoint',
                      'Oxygen': 'Treatment',
                      'CellType': 'celltype'},
                     inplace = True)

    adata.obs['celltype'].replace({'Cap':'Cap1','Cap-a':'Cap2'},inplace=True)
    adata.obs['timepoint_treatment'] =adata.obs['Timepoint'].astype(str) + '_' + adata.obs['Treatment'].astype(str)
    sc.pp.calculate_qc_metrics(
            adata,
            expr_type="counts",
            log1p=True,
            inplace=True,
        )
    ## read in cell cycle genes and score
    cell_cycle_genes = [x.strip() for x in open('/home/carsten/alvira_bioinformatics/venous_ec_scrnaseq/data/outside_data/regev_lab_cell_cycle_genes.txt')]
    cell_cycle_genes = [x.lower().capitalize() for x in cell_cycle_genes]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.pp.normalize_total(adata, key_added=None, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.score_genes(adata, ['Mki67', 'Top2a', 'Birc5', 'Hmgb2', 'Cenpf'], score_name='proliferation_score')
    adata.layers["log1p"] = adata.X.copy()


    # In[4]:


    endo_cts = ['Art','Cap1',
    'Vein',
    ]
    endo_adata = adata[adata.obs['celltype'].isin(endo_cts)].copy()
    sc.tl.score_genes_cell_cycle(endo_adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.pp.regress_out(endo_adata, ['S_score', 'G2M_score'])
    endo_adata.layers['cc_regress'] = endo_adata.X.copy()


    # In[5]:


    sc.pp.highly_variable_genes(endo_adata,batch_key='orig.ident')
    sc.pp.pca(endo_adata, mask_var="highly_variable")
    sce.pp.harmony_integrate(endo_adata,'orig.ident',max_iter_harmony = 20)
    sc.pp.neighbors(endo_adata, use_rep="X_pca_harmony")
    sc.tl.leiden(endo_adata, key_added="leiden")
    endo_adata.obs['Cell Subtype'] = endo_adata.obs['leiden'].map({'0':'Cap1','1':'Cap1','2':'Cap1','3':'Cap1','4':'Cap1','5':'Arterial EC','6':'Venous EC','7':'Cap1',})
    endo_adata.uns['Cell Subtype_colors']= ['#4A90E2','#9B59B6','#E35D6A']
    sc.tl.umap(endo_adata,min_dist=0.5)
    endo_adata.X = endo_adata.layers['log1p'].copy()
    start_and_end_states = {'root_ct':('Cap1',1,'min'),
                                'terminal_cts' :{'Arterial EC':(1,'max'),
                                                  'Venous EC':(0,'max')
                                                  }
                                }
    palantir_psuedotime_routine_external_datasets(endo_adata,start_and_end_states,save_prefix=figures,adata_name=adata_name,pca_key='X_pca')
    find_gene_overlap_in_pseudotimes(endo_adata,save_prefix=figures,adata_name=adata_name)

    sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5,1.5))
    sc.settings.figdir = figures
    sns.set_style('white', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    plt.rcParams["font.family"] = "Arial"
    adata_ven = endo_adata[endo_adata.obs['Cell Subtype']=='Venous EC']
    tp='P3'
    adata_ven_tp = adata_ven[adata_ven.obs['Timepoint'] == tp]
    adata_ven_tp.obs['Treatment'] = adata_ven_tp.obs['Treatment'].str[0]
    custom_dotplot(adata_ven_tp[adata_ven_tp.obs['Vessel size category'] != 'capillary'],
                   ['Mki67', 'Top2a', 'Birc5', 'Hmgb2', 'Cenpf'],
                   scale_by_gene=True,
                   x_obs='Treatment',
                   y_obs='Vessel size category',
                   x_order=['N', 'H'],
                   cmap='viridis',
                   dot_max_size=100,
                   figsize=(4.5, 2),
                   save=f'{figures}/dotplot_proliferation_vec_{tp}.png')

    sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1, 1))
    sc.settings.figdir = figures
    sns.set_style('white', rc={
        'xtick.bottom': True,
        'ytick.left': True,
    })
    plt.rcParams["font.family"] = "Arial"
    size = 15
    for gene in ['Cxcl12', 'Cxcr4', 'Ackr3', 'Esr2', 'Dkk2', 'Moxd1']:
        sc.pl.umap(endo_adata, color=[gene], cmap='viridis', size=size, frameon=False, save=f'hurskainen_{gene}.png')
    ls = []
    for x, y in zip(endo_adata.obs['Cell Subtype'], endo_adata.obs['Vessel size category']):
        if x == 'Cap1':
            ls.append('Cap1')
        else:
            if y == 'capillary':
                ls.append(f'Cap1')
                continue
            if x == 'Arterial EC':
                x = 'PAEC'
            else:
                x = 'PVEC'
            ls.append(f'{x} {y[0].upper()}')
    endo_adata.obs['ct_s'] = ls
    sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5, 1.5))

    gene_ls = [
        'Ccdc85a',
        'Glp1r',
        'Kit',
        'Sox4', 'Nrp1', 'Ifitm3',
        'Ptprr', 'Adgrg6', 'Foxo1',
        'Mgp', 'Eln', 'Nr4a2',
    ][::-1]
    sc.pl.dotplot(endo_adata, gene_ls, groupby='ct_s', cmap='viridis',
                  categories_order=['PAEC L', 'PAEC M', 'PAEC S', 'Cap1', 'PVEC S', 'PVEC M', 'PVEC L'],
                  standard_scale='var', save='size_axis.png')
    sc.pl.umap(endo_adata, color=gene_ls, hspace=0.5, wspace=0.3, cmap='viridis', save='size_axis.png')

    # In[31]:

    gene_ls = ['Dkk2', 'Sox6',
               'Lama3', 'Bdkrb2',
               # 'Depp1', not present
               'Stc1',

               'Adam23', 'Ntrk2',
               'Emid1', 'Chrm2',
               'Chrm3', 'Rarb',
               # 'Fads2b', not present
               'Gria3',
               'Ptger3', 'Moxd1',
               ]
    sc.pl.MatrixPlot(endo_adata, gene_ls, groupby='ct_s',
                     categories_order=['PAEC L', 'PAEC M', 'PAEC S', 'Cap1', 'PVEC S', 'PVEC M', 'PVEC L'],
                     standard_scale='var').style(cmap='viridis').swap_axes().savefig(
        f'{figures}/matrixplot_size_markers.png', dpi=300, bbox_inches='tight')
    sc.pl.DotPlot(endo_adata, gene_ls, groupby='ct_s',
                  categories_order=['PAEC L', 'PAEC M', 'PAEC S', 'Cap1', 'PVEC S', 'PVEC M', 'PVEC L'],
                  standard_scale='var').style(cmap='viridis').savefig(f'{figures}/matrixplot_size_markers.png', dpi=300,
                                                                      bbox_inches='tight')

    # for tp in ['P3','P7','P14']:
    #     tp_adata = endo_adata[endo_adata.obs['Timepoint']==tp]
    #     tp_adata.X = tp_adata.layers['cc_regress']
    #     figures_tp = f"data/figures/figures/hurskainen/{tp}"
    #     os.makedirs(figures_tp, exist_ok=True)
    #     sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5,1.5))
    #     sc.settings.figdir = figures_tp
    #     sc.pp.highly_variable_genes(tp_adata,batch_key='orig.ident')
    #     sc.pp.pca(tp_adata, mask_var="highly_variable")
    #     sce.pp.harmony_integrate(tp_adata,'orig.ident',max_iter_harmony = 20)
    #     sc.pp.neighbors(tp_adata, use_rep="X_pca")
    #     sc.tl.leiden(tp_adata, key_added="leiden")
    #     sc.tl.umap(tp_adata,min_dist=1)
    #     tp_adata.X = tp_adata.layers['log1p']
    #     sc.tl.rank_genes_groups(tp_adata,'leiden',method='wilcoxon')
    #     sc.pl.rank_genes_groups_dotplot(tp_adata,dendrogram=False)
    #     for color in ['celltype','Cell Subtype','Timepoint','Treatment','leiden','Gja5','Slc6a2','Kit','Hey1','Nr2f2','Mecom','Eln','Mgp','Col4a1','Col4a2']:
    #         sc.pl.umap(tp_adata,color=color,use_raw=False,cmap='viridis',size=size,frameon=False,save=f'{tp}_{color}.png')
    #     for treat in ['Normoxia','Hyperoxia']:
    #         treat_adata = tp_adata[tp_adata.obs['Treatment']==treat]
    #         treat_adata.X = treat_adata.layers['cc_regress']
    #         figures_tp = f"data/figures/figures/hurskainen/{tp}/{treat}"
    #         os.makedirs(figures_tp, exist_ok=True)
    #         sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5,1.5))
    #         sc.settings.figdir = figures_tp
    #         sc.pp.highly_variable_genes(treat_adata,batch_key='orig.ident')
    #         sc.pp.pca(treat_adata, mask_var="highly_variable")
    #         sce.pp.harmony_integrate(treat_adata,'orig.ident',max_iter_harmony = 20)
    #         sc.pp.neighbors(treat_adata, use_rep="X_pca")
    #         sc.tl.leiden(treat_adata, key_added="leiden")
    #
    #         sc.tl.umap(treat_adata,min_dist=1)
    #         treat_adata.X = treat_adata.layers['log1p']
    #         sc.tl.rank_genes_groups(treat_adata,'leiden',method='wilcoxon')
    #         sc.pl.rank_genes_groups_dotplot(treat_adata,dendrogram=False)
    #         for color in ['celltype','Cell Subtype','Timepoint','Treatment','leiden','Gja5','Slc6a2','Kit','Hey1','Nr2f2','Mecom','Eln','Mgp','Col4a1','Col4a2']:
    #             sc.pl.umap(treat_adata,color=color,use_raw=False,cmap='viridis',size=size,frameon=False,save=f'{tp}_{color}.png')
    #     for treat in ['Normoxia','Hyperoxia']:
    #         treat_adata = endo_adata[endo_adata.obs['Treatment']==treat]
    #         treat_adata.X = treat_adata.layers['cc_regress']
    #         figures_tp = f"data/figures/figures/hurskainen/{treat}"
    #         os.makedirs(figures_tp, exist_ok=True)
    #         sc.set_figure_params(dpi_save=300, fontsize=10, figsize=(1.5,1.5))
    #         sc.settings.figdir = figures_tp
    #         sc.pp.highly_variable_genes(treat_adata,batch_key='orig.ident')
    #         sc.pp.pca(treat_adata, mask_var="highly_variable")
    #         sce.pp.harmony_integrate(treat_adata,'orig.ident',max_iter_harmony = 20)
    #         sc.pp.neighbors(treat_adata, use_rep="X_pca")
    #         sc.tl.leiden(treat_adata, key_added="leiden")
    #         sc.tl.umap(treat_adata,min_dist=1)
    #         tp_adata.X = tp_adata.layers['log1p']
    #         sc.tl.rank_genes_groups(treat_adata,'leiden',method='wilcoxon')
    #         sc.pl.rank_genes_groups_dotplot(treat_adata,dendrogram=False)
    #         for color in ['celltype','Cell Subtype','Timepoint','Treatment','leiden','Gja5','Slc6a2','Kit','Hey1','Nr2f2','Mecom','Eln','Mgp','Col4a1','Col4a2']:
    #             sc.pl.umap(treat_adata,color=color,use_raw=False,cmap='viridis',size=size,frameon=False,save=f'{tp}_{color}.png')


