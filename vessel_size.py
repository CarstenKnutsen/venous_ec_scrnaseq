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
import seaborn as sns
import os
from functions import plot_obs_abundance,compare_obs_values_within_groups_to_excel
from openpyxl import load_workbook
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



adata_name = "venous_ec"
output = "data/figures/vessel_size"
os.makedirs(output, exist_ok=True)
data = "data/single_cell_files/scanpy_files"
os.makedirs(output, exist_ok=True)
sc.set_figure_params(dpi=300, dpi_save=300, format="png")
sc.settings.figdir = output
sc.settings.autoshow = False

def normalize_dataframe(df):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-10, 10))
    # Fit the scaler on the data and transform each column
    df_normalized = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    return df_normalized

if __name__ == "__main__":
    adata = sc.read(f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad")
    vec_size_comp = pd.read_excel(f'data/figures/subcluster_no_cc/Venous EC/leiden_Venous EC_comparisons.xlsx',
                                  sheet_name='3 v 4', index_col=0)
    art_size_comp = pd.read_excel(f'data/figures/subcluster_no_cc/Arterial EC/leiden_Arterial EC_comparisons.xlsx',
                                  sheet_name='0 v 4', index_col=0)
    size_score = vec_size_comp['scores'] + -art_size_comp['scores']  # comparisons are in opposite directions
    size_score = size_score.sort_values()
    large_genes = size_score.tail(10).index.tolist()
    large_genes.reverse()
    small_genes = size_score.head(10).index.tolist()
    print(small_genes)
    print(large_genes)
    vec_prolif_comp = pd.read_excel(f'data/figures/subcluster_no_cc/Venous EC/leiden_Venous EC_comparisons.xlsx',
                                    sheet_name='0 v 2', index_col=0)
    prolif_genes = ['Mki67', 'Top2a', 'Birc5', 'Hmgb2', 'Cenpf']
    endo_adata = adata[adata.obs['Lineage'] == 'Endothelial']

    ## Plot size and proliferation genes
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial',
                    color='Cell Subtype',
                    save='_celltype')
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial',
                    color=large_genes,
                    save='_large_vessel_genes'
                    )
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial',
                    color=small_genes,
                    save='_small_vessel_genes'
                    )
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial',
                    color=prolif_genes,
                    save='_proliferation_genes')

    sc.tl.score_genes(endo_adata,large_genes,score_name='large_vessel_score')
    sc.tl.score_genes(endo_adata,small_genes,score_name='small_vessel_score')
    endo_adata.obs['vessel_size_score'] = endo_adata.obs['large_vessel_score'] - endo_adata.obs['small_vessel_score']
    for score in ['vessel_size_score', 'proliferation_score']:
        endo_adata.obs[score] = scaler.fit_transform(endo_adata.obs[[score]])
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial',
                    color='vessel_size_score',
                    save='_vessel_score')

    for ct in ['Arterial EC', 'Venous EC']:
        output_ct = f'{output}/{ct}'
        os.makedirs(output_ct, exist_ok=True)
        sc.settings.figdir = output_ct
        ct_adata = sc.read(f'data/figures/subcluster_no_cc/{ct}/{ct}_adata.gz.h5ad')
        for score in ['vessel_size_score','proliferation_score']:
            ct_adata.obs[score] = endo_adata.obs[score]
        sc.pl.umap(ct_adata, color=f'leiden_{ct}')
        sc.pl.umap(ct_adata, color=prolif_genes,save=f'_{ct}_proliferation_genes')
        sc.pl.umap(ct_adata, color=large_genes,save=f'_{ct}_large_vessel_genes')
        sc.pl.umap(ct_adata, color=small_genes,save=f'_{ct}_small_vessel_genes')
        sc.pl.umap(ct_adata, color=['vessel_size_score','proliferation_score'],save=f'_{ct}_scores')
        df = ct_adata.obs[['vessel_size_score', 'proliferation_score', 'Treatment']].copy()
        df['vessel_size_score_bins'] = pd.cut(df['vessel_size_score'], bins=3,labels=['small','medium','large'])
        # Use barplot
        fig,ax =plt.subplots(1,1)
        ax = sns.barplot(data=df, x='vessel_size_score_bins', y='proliferation_score', ci=None)
        plt.xticks(rotation=45)
        fig.savefig(f'{output_ct}/barplot_{ct}_size_by_proliferation.png',dpi=300,bbox_inches='tight')
        plt.close()
        fig, ax = plt.subplots(1, 1)
        ax = sns.barplot(data=df, x='vessel_size_score_bins', y='proliferation_score', hue='Treatment',
                         hue_order=['Normoxia', 'Hyperoxia'], ci=None)
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.5, 1))
        plt.xticks(rotation=45)
        fig.savefig(f'{output_ct}/barplot_{ct}_size_by_proliferation_treatment.png',dpi=300,bbox_inches='tight')
        plt.close()
        fig, ax = plt.subplots(1, 1)
        ax = sns.violinplot(data=df, x='vessel_size_score_bins', y='vessel_size_score',hue='Treatment',hue_order=['Normoxia','Hyperoxia'], linewidth=0)
        fig.savefig(f'{output_ct}/violinplot_{ct}_vessel_size_score.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1)
        ax = sns.violinplot(data=df, x='Treatment', y='vessel_size_score', hue='Treatment',
                            hue_order=['Normoxia', 'Hyperoxia'], linewidth=0)
        fig.savefig(f'{output_ct}/violinplot_{ct}_vessel_size_score_treatment.png', dpi=300, bbox_inches='tight')
        plt.close()

        ct_adata.obs['vessel_size_score_bins'] = df['vessel_size_score_bins']
        compare_obs_values_within_groups_to_excel(ct_adata, 'Treatment', group_column='vessel_size_score_bins',
                                                  output_prefix=f"{output_ct}/hyperoxia_degs_by_size")
        degs = pd.read_excel(f"{output_ct}/hyperoxia_degs_by_size.xlsx", sheet_name=None, index_col=0, header=0)

        hyperoxia_score = pd.DataFrame(index=degs['small'].index)
        hyperoxia_score['small'] = degs['small']['scores']
        hyperoxia_score['medium'] = degs['medium']['scores']
        hyperoxia_score['large'] = degs['large']['scores']


        hyperoxia_score = normalize_dataframe(hyperoxia_score)
        hyperoxia_score['large_small_difference'] = hyperoxia_score['large'] - hyperoxia_score['small']
        hyperoxia_score = hyperoxia_score.sort_values('large_small_difference')
        with pd.ExcelWriter(
                f"{output_ct}/hyperoxia_degs_by_size.xlsx",
                mode="a",
                engine="openpyxl",
                if_sheet_exists="replace",
        ) as writer:
            hyperoxia_score.to_excel(writer, sheet_name='normalized_scores_together')
        fig, ax = plt.subplots(1, 1)
        ax = sns.scatterplot(data=hyperoxia_score, x='small', y='large', linewidth=0)
        fig.savefig(f'{output_ct}/scatterplot_{ct}_hyperoxia_degs_by_size_comparison.png',dpi=300,bbox_inches='tight')
        plt.close()
        size_degs = hyperoxia_score.head(5).index.tolist() + hyperoxia_score.tail(5).index.tolist()
        sc.pl.umap(ct_adata,color=size_degs,save=f'_{ct}_size_degs')
        sc.pl.dotplot(ct_adata, size_degs, groupby=['vessel_size_score_bins','Treatment'],save=f'{ct}_size_degs')
        plot_obs_abundance(ct_adata, f"vessel_size_score_bins", hue="Treatment", ordered=True, as_percentage=True,
                           save=f'{output_ct}/barplot_{ct}_size_treatment_abundance.png', hue_order=['Normoxia', 'Hyperoxia'])
        ct_adata.write(f'data/figures/subcluster_no_cc/{ct}/{ct}_adata_sized.gz.h5ad')
