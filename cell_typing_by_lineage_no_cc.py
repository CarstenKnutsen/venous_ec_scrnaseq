"""Goal:Cell typing by lineage
Date:241231
Author: Carsten Knutsen
conda_env:dendritic
"""

import pandas as pd
import os
import scanpy as sc
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import median_abs_deviation
import scanpy.external as sce
import itertools
import string
from gtfparse import read_gtf
import anndata
from collections import defaultdict
from functions import plot_obs_abundance
#we set hardcoded paths here
data = "data/single_cell_files/scanpy_files"
adata_name = "venous_ec"
figures = "data/figures/cell_typing_no_cc"
os.makedirs(figures, exist_ok=True)
sc.settings.figdir = figures
sc.set_figure_params(dpi=300, format="png")

gene_dict = {
        "mesenchymal": [
            "Col3a1",
            "G0s2",
            "Limch1",
            "Col13a1",
            "Col14a1",
            "Serpinf1",
            "Pdgfra",
            "Scara5",
            "Acta2",
            "Hhip",
            "Fgf18",
            "Wif1",
            "Tgfbi",
            "Tagln",
            "Mustn1",
            "Aard",
            "Pdgfrb",
            "Notch3",
            "Dcn",
            "Cox4i2",
            "Higd1b",
            "Wt1",
            "Lrrn4",
            "Upk3b",
            "Mki67",
            "Acta1",
            'Lgals3',
            # 'Tubb3',
            # 'Aqp3',
'Ttn','Ryr2','Myh6','Tbx20','Ldb3','Eya4','Rbm20','Neb','Itm2a','Mybpc1',
            'Col1a1',
            "Epcam",
            "Ptprc",
            "Pecam1",
            'Zbtb16',
        ],
        "endothelial": [
            "Gja5",
            "Dkk2",
            'Apln',
            'Aplnr',
            "Bmx",
            "Fn1",
            "Ctsh",
            "Kcne3",
            "Cdh13",
            "Car8",
            "Mmp16",
            "Slc6a2",
            "Thy1",
            "Mmrn1",
            "Ccl21a",
            "Reln",
            "Neil3",
            "Mki67",
            "Aurkb",
            "Depp1",
            "Ankrd37",
            "Peg3",
            "Mest",
            "Hpgd",
            "Cd36",
            "Car4",
            "Sirpa",
            "Fibin",
            'Tbx2',
            "Col15a1",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
            "Cdh5",
        ],
        "immune": [
            "Cd68",
            "Gal",
            "Itgax",
            "Car4",
            "C1qa",
            "Plac8",
            "Batf3",
            "Itgae",
            # "Cd209a",
            "Mreg",
            "Mcpt4",
            "Mcpt8",
            "Retnlg",
            "Ms4a1",
            "Gzma",
            "Cd3e",
            "Areg",
            "Mki67",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
        ],
        "epithelial": [
            'Muc1',
            "Scg5",
            "Ascl1",
            "Lyz1",
            "Lyz2",
            "Sftpc",
            "Slc34a2",
            "S100g",
            "Sftpa1",
            "Akap5",
            "Hopx",
            "Col4a4",
            "Vegfa",
            "Lyz1",
            "Tmem212",
            "Dynlrb2",
            "Cdkn1c",
            "Tppp3",
            "Scgb3a2",
            "Cyp2f2",
            "Scgb1a1",
            "Reg3g",
            "Scgb3a1",
            "Mki67",
            "Col1a1",
            "Epcam",
            "Ptprc",
            "Pecam1",
            'Dnah12',
            'Spag17',
            'Muc5b',

        ],
    }
leiden_ct_dict = {
    "Mesenchymal": {
        "0": "Alveolar fibroblast",
        "1": "Adventitial fibroblast",
        "2": "Mesothelial",
        "3": "Airway smooth muscle",
        "4": "Pericyte",
        "5": "Vascular smooth muscle",
        "6": "Myofibroblast",
        "7": "low-quality_Fibroblast",
        "8":"Adventitial fibroblast",
        "9": "Schwann cell",
        "10":  "Alveolar fibroblast",
        "11":"doublet_Epithelial",
        "12":"low-quality Pericyte",
        "13":"Mesothelial",
        "14": "Mesothelial",
        "15": "Aberrant muscle",
        "16":"Striated muscle",
        "17": "Dsc2_3+ cell",
        "18": "Airway smooth muscle"

    },
    "Endothelial": {
        "0": "Cap1",
        "1": "Cap1_Cap2",
        "2": "Venous EC",
        "3": "Arterial EC",
        "4": "Lymphatic EC",
        "5": "Cap2",
        "6": "Venous EC",
        "7": "Venous EC",
        "8":  "Systemic Venous EC",
        "9":  "doublet_Arterial EC",
        "10": "low-quality_Arterial EC",
        "11": "low-quality_Lymphatic EC",
        "12": "Venous EC",
    },
    "Immune": {
        "0": "Alveolar macrophage",
        "1": "Alveolar macrophage",
        "2": "Alveolar macrophage",
        "3": "B cell",
        "4": "doublet_endothelial",
        "5": "Monocyte",
        "6": "Alveolar macrophage",
        "7": "Mast cell",
        "8": "Alveolar macrophage",
        "9": "Interstitial macrophage",
        "10":  "Alveolar macrophage",
        "11": "B cell",
        "12":"Alveolar macrophage",
        "13": "Basophil",
        "14": "T cell",
        "15": "c-Dendritic cell",
        "16": "mig-Dendritic cell",
    },
    "Epithelial": {
        "0": "AT1_AT2",
        "1": "Ciliated",
        "2": "AT2",
        "3": "AT1",
        "4": "Ciliated",
        "5": "AT2",
        "6": "Club",
        "7": "Goblet",
        "8": "doublet_epithelial cell",
        "9": "AT1",
        "10": "Ciliated",
        "11": "Neuroendocrine",
        "12": "Ciliated",

    },

}
if __name__ == "__main__":
    adata = sc.read(
        f"{data}/{adata_name}_filtered_embed.gz.h5ad",
    )
    print(adata)
    adata.obs["Cell Subtype"] = pd.Series(index=adata.obs.index, data=None, dtype="str")
    cell_cycle_genes = [x.strip() for x in open('data/outside_data/regev_lab_cell_cycle_genes.txt')]
    cell_cycle_genes = [x.lower().capitalize() for x in cell_cycle_genes]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.tl.score_genes(adata, ['Mki67', 'Top2a', 'Birc5', 'Hmgb2', 'Cenpf'], score_name='proliferation_score')
    for lineage in adata.obs['Lineage'].cat.categories:
        figures_lin = f"{figures}/{lineage}"
        os.makedirs(figures_lin, exist_ok=True)
        sc.settings.figdir = figures_lin
        print(lineage)
        lin_adata = adata[adata.obs['Lineage'] == lineage]
        sc.pp.regress_out(lin_adata, ['S_score', 'G2M_score'])
        sc.pp.highly_variable_genes(lin_adata, batch_key="Library")
        sc.pp.pca(lin_adata, mask_var="highly_variable")
        sc.pp.neighbors(lin_adata, use_rep="X_pca")
        if lineage =='Epithelial':
            sc.tl.leiden(lin_adata, key_added=f"leiden_{lineage}", resolution=0.75) # need clustering for neuroendocrine to split
        else:
            sc.tl.leiden(lin_adata, key_added=f"leiden_{lineage}", resolution=0.5)
        sc.tl.umap(lin_adata, min_dist=0.5)
        sc.tl.dendrogram(lin_adata,f"leiden_{lineage}")
        lin_adata.X = lin_adata.layers['log1p'].copy()
        sc.tl.rank_genes_groups(lin_adata, groupby=f"leiden_{lineage}",method='wilcoxon',pts=True)
        sc.pl.rank_genes_groups_dotplot(
            lin_adata,
            groupby=f"leiden_{lineage}",
            show=False,
            save=f"leiden_markers.png",
        )
        with pd.ExcelWriter(
                f"{figures_lin}/{lineage}_leiden_markers.xlsx", engine="xlsxwriter"
        ) as writer:
            for ct in lin_adata.obs[f"leiden_{lineage}"].cat.categories:
                df = sc.get.rank_genes_groups_df(lin_adata, key="rank_genes_groups", group=ct)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
        sc.pl.umap(lin_adata, color = ['leiden',f"leiden_{lineage}",'celltype_rough'],wspace=0.5,show=False,save='_pretrim_leiden')
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby=f"leiden_{lineage}",show=False,save='useful_genes_leiden')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_pretrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis'],wspace=0.5,show=False,save='_pretrim_qc')
        if lineage == 'Mesenchymal':
            weird_cells = {'striated_muscle': ['Ttn', 'Ryr2', 'Myh6', 'Tbx20', 'Ldb3'],
                         'multi_ab_musc': ['Gm20754', 'Pdzrn4', 'Chrm2', 'Cacna2d3', 'Chrm3'],
                         'multi_acta1': ['Eya4', 'Rbm20', 'Neb', 'Itm2a', 'Mybpc1'],
                         'male hyperoxic fibroblast': ['Acta1', 'Actc1', 'Tubb3', 'Tuba4a'],
                         'male hyperoxic mystery': ['Acta1', 'Eif1', 'Tuba1c', 'Emd']

                         }
            sc.pl.dotplot(lin_adata, weird_cells, groupby=f"leiden_{lineage}", show=False,
                          save='weird_cells')
        lin_adata.obs["Cell Subtype"] = [leiden_ct_dict[lineage][x] for x in lin_adata.obs[f"leiden_{lineage}"]]
        lin_adata = lin_adata[(~lin_adata.obs["Cell Subtype"].str.startswith('doublet')) & (~lin_adata.obs["Cell Subtype"].str.startswith('low-quality'))]
        sc.tl.umap(lin_adata,min_dist=0.5)
        sc.pl.umap(lin_adata, color = ['Treatment','Library','leiden',f"leiden_{lineage}",'celltype_rough',"Cell Subtype"],wspace=0.5,show=False,save='_posttrim_leiden')
        sc.pl.dotplot(lin_adata,gene_dict[lineage.lower()],groupby="Cell Subtype",show=False,save='useful_genes_celltype')
        sc.pl.umap(lin_adata, color = gene_dict[lineage.lower()],wspace=0.5,show=False,save='_posttrim_genes')
        sc.pl.umap(lin_adata, color = ['log1p_total_umis','log1p_n_genes_by_umis'],wspace=0.5,show=False,save='_posttrim_qc')
        sc.pl.umap(lin_adata, color = ['phase','proliferation_score'],wspace=0.5,show=False,save='_posttrim_cc')
        sc.tl.rank_genes_groups(lin_adata, groupby="Cell Subtype", method='wilcoxon', pts=True)
        sc.tl.dendrogram(lin_adata,"Cell Subtype")
        sc.pl.rank_genes_groups_dotplot(
        lin_adata,
        groupby = "Cell Subtype",
        show = False,
        save = f"celltype_markers.png",
        )
        with pd.ExcelWriter(
            f"{figures_lin}/{lineage}_celltype_markers.xlsx", engine = "xlsxwriter"
        ) as writer:
            for ct in lin_adata.obs["Cell Subtype"].cat.categories:
                df = sc.get.rank_genes_groups_df(lin_adata, key="rank_genes_groups", group=ct)
                df.set_index("names")
                df["pct_difference"] = df["pct_nz_group"] - df["pct_nz_reference"]
                df.to_excel(writer, sheet_name=f"{ct} v rest"[:31])
        # Add Lineage umaps and leiden clusters to top level
        adata.obs[f"umap_{lineage}_1"] = np.nan
        adata.obs[f"umap_{lineage}_2"] = np.nan
        lin_adata.obs[f"umap_{lineage}_1"] = [x[0] for x in lin_adata.obsm["X_umap"]]
        lin_adata.obs[f"umap_{lineage}_2"] = [x[1] for x in lin_adata.obsm["X_umap"]]
        adata.obs[f"umap_{lineage}_1"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"umap_{lineage}_1"
        ]
        adata.obs[f"umap_{lineage}_2"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"umap_{lineage}_2"
        ]
        adata.obs[f"leiden_{lineage}"] = np.nan
        adata.obs[f"leiden_{lineage}"].loc[lin_adata.obs.index] = lin_adata.obs[
            f"leiden_{lineage}"
        ]
        adata.obsm[f"X_umap_{lineage}"] = adata.obs[
            [f"umap_{lineage}_1", f"umap_{lineage}_2"]
        ].to_numpy()
        del adata.obs[f"umap_{lineage}_1"]
        del adata.obs[f"umap_{lineage}_2"]
        adata.obs["Cell Subtype"].loc[lin_adata.obs.index] = lin_adata.obs[
            "Cell Subtype"
        ]
        plot_obs_abundance(lin_adata,'Cell Subtype',hue='Treatment',ordered=True,
                       as_percentage=True,save=f"{figures_lin}/{lineage}_celltype_abundance.png",hue_order=['Normoxia','Hyperoxia'])
    adata = adata[~adata.obs['Cell Subtype'].isna()]
    ct_order = []
    for lin in adata.obs['Lineage'].cat.categories:
        for ct in sorted(adata[adata.obs['Lineage'] == lin].obs['Cell Subtype'].unique()):
            ct_order.append(ct)
    sc.tl.umap(adata,min_dist=0.5)
    adata.obs['Cell Subtype'] = pd.Categorical(adata.obs['Cell Subtype'], categories=ct_order)
    sc.settings.figdir = figures
    sc.pl.umap(adata,color='Cell Subtype',save='Cell_Subtype',show=False)
    plot_obs_abundance(adata, 'Cell Subtype', hue='Treatment', ordered=True,
                       as_percentage=True, save=f"{figures}/celltype_abundance.png",hue_order=['Normoxia','Hyperoxia'])
    with pd.ExcelWriter(
        f"{figures}/celltype_counts.xlsx", engine="xlsxwriter"
    ) as writer:
        obs_list = ["Library", "Treatment", "Cell Subtype"]
        num_obs = len(obs_list) + 1
        for ind in range(0, num_obs):
            for subset in itertools.combinations(obs_list, ind):
                if len(subset) != 0:
                    subset = list(subset)
                    if len(subset) == 1:
                        key = subset[0]
                        adata.obs[key].value_counts().to_excel(writer, sheet_name=key)
                    else:
                        key = "_".join(subset)
                        adata.obs.groupby(subset[:-1])[subset[-1]].value_counts(
                            normalize=True
                        ).to_excel(writer, sheet_name=key[:31])
    adata.write(
        f"{data}/{adata_name}_celltyped_no_cc.gz.h5ad", compression="gzip"
    )

