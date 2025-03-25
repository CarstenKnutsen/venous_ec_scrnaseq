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
                                  sheet_name='1 v 4', index_col=0)

    size_score = normalize_dataframe(pd.DataFrame(vec_size_comp['scores'])) + normalize_dataframe(
        pd.DataFrame(art_size_comp['scores']))
    size_score = size_score.sort_values('scores')['scores']
    large_genes = size_score.head(10).index.tolist()
    large_genes.reverse()
    small_genes = size_score.tail(10).index.tolist()
    print(small_genes)
    print(large_genes)
    prolif_genes = ['Mki67', 'Top2a', 'Birc5', 'Hmgb2', 'Cenpf']
    endo_adata = adata[adata.obs['Lineage'] == 'Endothelial']
    ## Plot size and proliferation genes
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial_no_cc',
                    color='Cell Subtype',
                    save='_celltype')
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial_no_cc',
                    color='Cell Subtype_no_cc',
                    save='_celltype_no_cc')
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial_no_cc',
                    color=large_genes,
                    save='_large_vessel_genes'
                    )
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial_no_cc',
                    color=small_genes,
                    save='_small_vessel_genes'
                    )
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial_no_cc',
                    color=prolif_genes,
                    save='_proliferation_genes')

    sc.tl.score_genes(endo_adata,large_genes,score_name='large_vessel_score')
    sc.tl.score_genes(endo_adata,small_genes,score_name='small_vessel_score')
    endo_adata.obs['vessel_size_score'] = endo_adata.obs['large_vessel_score'] - endo_adata.obs['small_vessel_score']
    for score in ['vessel_size_score', 'proliferation_score']:
        endo_adata.obs[score] = scaler.fit_transform(endo_adata.obs[[score]])
    sc.pl.embedding(endo_adata,
                    basis='X_umap_Endothelial_no_cc',
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
        ax = sns.histplot(data=df, x="vessel_size_score", hue='Treatment',hue_order=['Normoxia','Hyperoxia'], stat='probability',
                              element='poly', fill=False, common_norm=False, bins=10, ax=ax)
        fig.savefig(f'{output_ct}/histplot_{ct}_vessel_size_score.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1)
        ax = sns.histplot(data=df, x="proliferation_score", hue='Treatment', hue_order=['Normoxia', 'Hyperoxia']
        ,stat = 'probability',
        element = 'poly', fill = False, common_norm = False, bins = 10, ax = ax)
        fig.savefig(f'{output_ct}/histplot_{ct}_proliferation_score.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, axs = plt.subplots(3, 1,figsize=(3,9),sharex=True)
        axs = axs.ravel()
        df['size'] = df['vessel_size_score_bins']
        for i,size in enumerate(['small','medium','large']):
            ax = sns.histplot(data=df[df['size']==size], x="proliferation_score", hue='Treatment', hue_order=['Normoxia', 'Hyperoxia']
                              , stat='probability',
                              element='poly', fill=False, common_norm=False, bins=10, ax=axs[i])


            ax.set_title(size)
            if i!=2:
                ax.get_legend().remove()
        fig.tight_layout()
        fig.savefig(f'{output_ct}/histplot_{ct}_size_proliferation_score.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, axs = plt.subplots(2, 1,figsize=(3,6),sharex=True)
        axs = axs.ravel()
        for i,treat in enumerate(['Normoxia', 'Hyperoxia']):
            ax = sns.histplot(data=df[df['Treatment']==treat], x="proliferation_score", hue='size', hue_order=['small','medium','large']
                              , stat='probability',
                              element='poly', fill=False, common_norm=False, bins=10, ax=axs[i])
            if i==0:
                ax.get_legend().remove()
            ax.set_title(treat)
        fig.tight_layout()
        fig.savefig(f'{output_ct}/histplot_{ct}_treat_proliferation_score.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1)
        ax = sns.histplot(data=df, x="proliferation_score", y='vessel_size_score', hue='Treatment', hue_order=['Normoxia', 'Hyperoxia'], ax = ax)
        fig.savefig(f'{output_ct}/histplot_2d_{ct}_size_proliferation_treatment.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1)
        ax = sns.histplot(data=df, x="proliferation_score", y='vessel_size_score', ax=ax)
        fig.savefig(f'{output_ct}/histplot_2d_{ct}_size_proliferation.png', dpi=300, bbox_inches='tight')
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

#### VSM
    vsm_size_comp = pd.read_excel(f'data/figures/subcluster_no_cc/Vascular smooth muscle/leiden_Vascular smooth muscle_comparisons.xlsx', sheet_name='1 v 2',
                             index_col=0, header=0)

    output_ct = f'{output}/mural'
    os.makedirs(output_ct, exist_ok=True)
    sc.settings.figdir = output_ct
    mural_adata = adata[adata.obs['Cell Subtype_no_cc'].isin(['Vascular smooth muscle', 'Pericyte'])]

    sc.tl.score_genes(mural_adata, vsm_size_comp.head(10).index.tolist(), score_name='large_vessel_score')
    sc.tl.score_genes(mural_adata, vsm_size_comp.tail(10).index.tolist(), score_name = 'small_vessel_score')
    mural_adata.obs['vessel_size_score'] = mural_adata.obs['large_vessel_score'] - mural_adata.obs['small_vessel_score']
    for score in ['vessel_size_score', 'proliferation_score']:
        mural_adata.obs[score] = scaler.fit_transform(mural_adata.obs[[score]])
    sc.pp.highly_variable_genes(mural_adata,
                                batch_key="Library"
                                )
    sc.pp.pca(mural_adata, use_highly_variable=True)

    sc.pp.neighbors(mural_adata, use_rep='X_pca')
    sc.tl.leiden(
        mural_adata,
        key_added=f"leiden_subcluster",
        resolution=0.5
    )
    sc.tl.umap(mural_adata, min_dist=0.5)
    sc.tl.rank_genes_groups(mural_adata, f"leiden_subcluster", pts=True, method="wilcoxon")
    sc.pl.umap(mural_adata, color=['Cell Subtype', 'Cell Subtype_no_cc', 'leiden_subcluster', 'leiden_Mesenchymal',
                                   'proliferation_score'], wspace=0.6, save='mural_metadata.png')
    sc.pl.umap(mural_adata, color=vsm_size_comp.head(20).index.tolist(), wspace=0.6,
               save='mural_large_genes.png')
    sc.pl.umap(mural_adata, color=vsm_size_comp.tail(20).index.tolist(), wspace=0.6,
               save='mural_small_genes.png')
    mural_dt = {'0': 'VSM',
                '1': 'Per',
                '2': 'VSM',
                '3': 'Int',
                '4': 'ProVSM',
                '5': 'VSM',
                '6': 'ProPer_Int',
                '7': 'ProVSM',
                }
    mural_adata.obs['Mural Subtype'] = [mural_dt[x] for x in mural_adata.obs['leiden_subcluster']]
    compare_obs_values_within_groups_to_excel(mural_adata,'Mural Subtype',output_prefix=f'{output_ct}/mural_subtype_comparison')
    sc.pl.umap(mural_adata, color=['leiden_subcluster', 'Cell Subtype_no_cc', 'proliferation_score', 'Mural Subtype','vessel_size_score'],
               save='mural_metadata.png', ncols=2)
    sc.pl.dotplot(mural_adata, vsm_size_comp.head(20).index.tolist() + vsm_size_comp.tail(20).index.tolist(),
                  groupby='Mural Subtype', save='mural_subtype_large_small.png')
    sc.pl.umap(mural_adata, color=['Eln', 'Eng'], save='mural_eln_eng.png', ncols=2)
    vsm_overlap_large = [x for x in vsm_size_comp['scores'].sort_values().tail(100).index if
                         x in size_score.head(100).index]
    vsm_overlap_small = [x for x in vsm_size_comp['scores'].sort_values().head(100).index if
                         x in size_score.tail(100).index]
    for gene_ls in [vsm_overlap_large, vsm_overlap_small]:
        for gene in gene_ls:
            fig, axs = plt.subplots(1, 2, figsize=(5, 2))
            axs = axs.ravel()
            sc.pl.embedding(adata[adata.obs['Lineage'] == 'Endothelial'],
                            basis='X_umap_Endothelial_no_cc',
                            color=gene,
                            ax=axs[0],
                            frameon=False,
                            show=False
                            )
            sc.pl.umap(mural_adata,
                       color=gene,
                       ax=axs[1],
                       frameon=False,

                       show=False

                       )
            fig.savefig(f'{output_ct}/umap_size_mural_endo_overlap_{gene}.png', dpi=300, bbox_inches='tight')
            plt.close()
    sc.pl.dotplot(adata, vsm_overlap_large + vsm_overlap_small, groupby='Cell Subtype_no_cc',
                  save='overlap_size_genes_all_cts.png')
    df_3 = sc.get.rank_genes_groups_df(mural_adata,group='3')
    df_3['difference'] = df_3['pct_nz_group'] - df_3['pct_nz_reference']
    sc.pl.umap(mural_adata,color=df_3.sort_values('difference',ascending=False).head(20)['names'].values,save='int_specific_genes.png')
    genes = df_3.sort_values('difference',ascending=False).head(50)['names'].values
    sc.pl.dotplot(adata,df_3.sort_values('difference',ascending=False).head(20)['names'].values,groupby='Cell Subtype_no_cc',save='int_specific_genes.png')
    mural_subtype_comp = pd.read_excel('data/pilot/250318_vsm_deep_dive/mural_subtype_comparison.xlsx',
                                       sheet_name=['Int v Per', 'Int v VSM'], index_col=0, header=0)
    ct1 = 'Per'
    ct2 = 'VSM'
    score_df = pd.DataFrame(index=mural_subtype_comp['Int v Per'].index)
    score_df['Per'] = mural_subtype_comp['Int v Per']['scores']
    score_df['VSM'] = mural_subtype_comp['Int v VSM']['scores']

    score_df['sum'] = score_df[ct1] + score_df[ct2]
    score_df = score_df.sort_values('sum', ascending=False)
    score_df = score_df.loc[(~score_df.index.str.startswith('mt')) & (~score_df.index.str.startswith('Rps')) & (
        ~score_df.index.str.startswith('Rpl'))]
    fig,ax = plt.subplots()
    sns.scatterplot(data=score_df, x=ct1, y=ct2, linewidth=0,ax=ax)

    for gene in score_df.head(10).index:
        plt.text(
            score_df[ct1][gene],
            score_df[ct2][gene],
            gene,
            size=10,
        )
    for gene in score_df.tail(10).index:
        plt.text(
            score_df[ct1][gene],
            score_df[ct2][gene],
            gene,
            size=10,
        )
    fig.savefig(f'{output_ct}/scatterplot_int_v_vsmorper.png', dpi=300, bbox_inches='tight')
    plt.close()
    genes = score_df.head(10).index
    sc.pl.umap(mural_adata,color=genes,save='int_specific_genes.png')
    sc.pl.dotplot(adata,genes,groupby='Cell Subtype_no_cc',save='int_specific_genes.png')
    notch_genes = ['Notch1', 'Notch2', 'Notch3', 'Dll1', 'Dll3', 'Dll4', 'Jag1', 'Jag2', 'Dlk1']
    sc.pl.umap(mural_adata, color=notch_genes)
    sc.pl.dotplot(mural_adata, notch_genes, groupby='Mural Subtype',
                  categories_order=['Per', 'ProPer_Int', 'Int', 'VSM', 'ProVSM', ], save='mural_notch.png')
    sc.pl.embedding(adata[adata.obs['Lineage'] == 'Endothelial'],
                    basis='X_umap_Endothelial_no_cc',
                    color=notch_genes)
    sc.pl.dotplot(
        adata[adata.obs['Cell Subtype_no_cc'].isin(['Arterial EC', 'Cap1', 'Cap1_Cap2', 'Cap2', 'Venous EC', ])],
        notch_genes,
        categories_order=['Arterial EC', 'Venous EC', 'Cap1', 'Cap1_Cap2', 'Cap2'],
        groupby='Cell Subtype_no_cc',
        save='endo_nothch.png')
    endo_adata = endo_adata[endo_adata.obs['Cell Subtype_no_cc'].isin(['Arterial EC', 'Cap1', 'Cap1_Cap2', 'Cap2', 'Venous EC', ])]
    endo_adata.obs['size'] =pd.cut(endo_adata.obs['vessel_size_score'], bins=3,labels=['small','medium','large'])
    ls = []
    for x, y in zip(endo_adata.obs['Cell Subtype_no_cc'],endo_adata.obs['size']):
        if x.startswith('Cap'):
            ls.append(x)
        else:
            ls.append(f'{x}_{y}')
    endo_adata.obs['Cell Subtype_size'] = ls

    mural_adata.obs['Cell Subtype_size'] = mural_adata.obs['Mural Subtype']

    vessel_adata = endo_adata.concatenate(mural_adata)
    vessel_adata.write(f'{data}/{adata_name}_vessel_size_adata.gz.h5ad',compression='gzip')



