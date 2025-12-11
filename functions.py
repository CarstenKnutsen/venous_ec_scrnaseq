import scanpy as sc
import pandas as pd
import numpy as np
import os
import json
import shutil
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib_venn import venn2,venn3
import palantir
import cellrank as cr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.lines as mlines


def read_adata(folder):
    adata = sc.read_mtx(f'{folder}/matrix.mtx').T
    features = pd.read_csv(f'{folder}/genes.tsv',
                          sep = '\t',
                          header = None)
    bc = pd.read_csv(f'{folder}/barcodes.tsv',
                          sep = '\t',
                    header = None)
    features.rename(columns={0:'gene_id',
                            1: 'gene_symbol',
                            2: 'category'},
                   inplace = True)

    adata.var = features
    adata.obs_names = bc[0]
    adata.var_names = adata.var['gene_id'].values
    return adata

def compare_obs_values_within_groups_to_excel(
    adata, obs_column, group_column=None, output_prefix="comparisons"
):
    """
    Perform pairwise comparisons for values in an AnnData `obs` column, optionally within groups
    defined by a second `obs` column, and save results to Excel files.

    If only two categories exist in the `obs_column` and a `group_column` is provided,
    a single Excel file is created with each group as a separate sheet.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    obs_column : str
        The `obs` column containing categorical values to compare.
    group_column : str, optional
        The `obs` column defining groups within which comparisons are performed. If None, all data is used.
    output_prefix : str
        Prefix for the output Excel files. Filenames will be formatted as `{output_prefix}_{group}.xlsx`.

    Returns:
    --------
    None
    """
    # Ensure obs_column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[obs_column]):
        adata.obs[obs_column] = adata.obs[obs_column].astype("category")
    # Get categories in the obs_column
    obs_categories = adata.obs[obs_column].cat.categories
    # Check if group_column is provided and ensure it is categorical
    if group_column:
        if group_column not in adata.obs.columns:
            raise ValueError(f"Column '{group_column}' not found in adata.obs.")
        if not pd.api.types.is_categorical_dtype(adata.obs[group_column]):
            adata.obs[group_column] = adata.obs[group_column].astype("category")
        groups = adata.obs[group_column].cat.categories
    else:
        groups = [None]



    if len(obs_categories) == 2 and group_column:
        # Special case: only two categories in obs_column with group_column
        category, reference = obs_categories[0], obs_categories[1]
        output_file = f"{output_prefix}.xlsx"
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for group in groups:
                subset = adata[adata.obs[group_column] == group].copy() if group else adata
                _compare_and_write_to_excel(
                    subset, obs_column, category, reference, writer, sheet_name=f"{group}" if group else "All"
                )
        print(f"Results saved to {output_file}")
    else:
        # General case: multiple categories or no group_column
        for group in groups:
            subset = adata[adata.obs[group_column] == group].copy() if group else adata
            output_file = f"{output_prefix}_{group}.xlsx" if group else f"{output_prefix}.xlsx"
            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                for i, category in enumerate(obs_categories):
                    for reference in obs_categories[i + 1:]:
                        _compare_and_write_to_excel(
                            subset,
                            obs_column,
                            category,
                            reference,
                            writer,
                            sheet_name=f"{category} v {reference}",
                        )
            print(f"Results saved to {output_file}")
def _compare_and_write_to_excel(adata, obs_column, category, reference, writer, sheet_name):
    """
    Perform a single pairwise comparison between two categories and write results to an Excel sheet.

    Parameters:
    -----------
    adata : AnnData
        Subsetted data containing only the two categories being compared.
    obs_column : str
        The `obs` column used for grouping.
    category : str
        The category to compare.
    reference : str
        The reference category.
    writer : pd.ExcelWriter
        The Excel writer object to save results.
    sheet_name : str
        The name of the sheet in the Excel file.

    Returns:
    --------
    None
    """
    # Subset to only the two categories being compared
    mask = adata.obs[obs_column].isin([category, reference])
    pairwise_subset = adata[mask].copy()

    # Update the group labels for the subset
    pairwise_subset.obs[obs_column] = pairwise_subset.obs[obs_column].cat.remove_unused_categories()

    # Run rank_genes_groups
    try:
        sc.tl.rank_genes_groups(
            pairwise_subset,
            groupby=obs_column,
            groups=[category],
            method="wilcoxon",
            pts=True,
        )

        # Extract results
        result = sc.get.rank_genes_groups_df(pairwise_subset, group=category)

        # Align genes in result['names'] with adata.var_names
        gene_order = pairwise_subset.var_names.get_indexer(result['names'])

        # Calculate percent expression for the reference group
        ref_mask = pairwise_subset.obs[obs_column] == reference
        ref_expr = pairwise_subset.X[ref_mask][:, gene_order]


        # Calculate average expression for the current and reference groups
        category_mask = pairwise_subset.obs[obs_column] == category
        category_expr = pairwise_subset.X[category_mask][:, gene_order]

        avg_expr_category = np.asarray(np.mean(category_expr, axis=0)).flatten()
        avg_expr_reference = np.asarray(np.mean(ref_expr, axis=0)).flatten()


        # Add average expression columns
        result[f"avg_expr_{category}"] = avg_expr_category
        result[f"avg_expr_{reference}"] = avg_expr_reference

        # Rename the `pct_nz` column for the category
        result.columns.values[5] = f"pct_nz_{category}"
        result.columns.values[6] = f"pct_nz_{reference}"


        # Write to Excel
        result.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    except:
        pass


def deg_to_metascape(df,
                         output_fol,
                     key,
                         msbio_fol = '/home/carsten/MSBio/msbio_v3.5.20230101/',
                         species='10090', # Takes taxonomy id 10090 is mouse
                     threshold = False,
                         fc_name='logFC',
                         fc_threshold=1,
                         pvalue_name='FDR',
                         pvalue_threshold=0.05):
    '''Takes a  pandas df with DEGs and runs metascape on up and downregulated genes from each sheet

    !!!! Makes a new folder. Will delete existing folder and all contents if run on with output_fol as existing folder
    '''

    ## tmp file paths for msbio
    key = key.replace(" ", "").replace(".","_")
    print('writing tmp files')
    unlock = f'sudo chown -R $USER ~/MSBio'
    sp.run(unlock, cwd = msbio_fol, shell=True)
    input_fol = f'{msbio_fol}data/tmp/input/'
    tmp_output_fol = f'{msbio_fol}data/tmp/output/'
    job_file = f'{input_fol}batch.job'
    for path in [input_fol, tmp_output_fol]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    ### write csv files for gene lists
    if threshold == True:
        df = df.loc[(abs(df[fc_name]) > fc_threshold) &
                    (df[pvalue_name] < pvalue_threshold)]
    df.loc[df[fc_name] > 0].index.to_series(name='Gene').to_csv(f'{input_fol}{key}_up.csv', index=False) ## write the upregulated genes tmp files
    df.loc[df[fc_name] < 0].index.to_series(name='Gene').to_csv(f'{input_fol}{key}_down.csv', index=False) ## write the downregulated genes tmp files
    ### make job file
    json_dict = {}
    for fn in os.listdir(input_fol):
        tmp_output_fol2 = f'{tmp_output_fol}{".".join(fn.split(".")[:-1])}'
        os.makedirs(tmp_output_fol2, exist_ok=True)
        tmp_dict = {}
        tmp_dict['input'] = f'/{"/".join(f"{input_fol}{fn}".split("/")[-4:])}'
        tmp_dict['output'] =f'/{"/".join(f"{tmp_output_fol2}".split("/")[-4:])}'
        json_dict[fn.split('.')[0]] = tmp_dict
    with open(job_file, "w") as outfile:
        outfile.write(
            '\n'.join(json.dumps(json_dict[k]) for k in sorted(json_dict.keys()))
        )
    ## run MSBIO
    print('running MSBio')
    # commands
    up = f'bin/up.sh'
    job_run = f'bin/ms.sh /data/tmp/input/batch.job -u -S {species} -T {species}'
    down = f'bin/down.sh'
    # run commands
    sp.call(up, cwd = msbio_fol, shell=True)
    sp.call(job_run, cwd = msbio_fol, shell=True)
    sp.call(down, cwd = msbio_fol, shell=True)
    sp.call(unlock, cwd = msbio_fol, shell=True)
    # copy to final location
    if os.path.exists(output_fol):
        shutil.rmtree(output_fol)
    shutil.copytree(tmp_output_fol, output_fol)
    for path in [input_fol, tmp_output_fol]:
        if os.path.exists(path):
            shutil.rmtree(path)




def plot_obs_abundance(adata, obs_column, hue=None, groupby_column=None, figsize=(10, 6), save=None, ordered=False,
                       as_percentage=False, **kwargs):
    """
    Creates a barplot showing abundances of values in an obs column of an AnnData object with optional grouping for error bars.

    Parameters:
        adata: AnnData
            An annotated data matrix.
        obs_column: str
            Column in `adata.obs` to count values from.
        hue: str, optional
            Column in `adata.obs` to group counts by for color coding (Seaborn's hue).
        groupby_column: str, optional
            Column to apply a grouping for separate counts and error bars (e.g., for generating error bars).
        figsize: tuple
            Size of the resulting plot.
        save: str, optional
            Path to save the plot as an image. If None, the plot is not saved.
        ordered: bool
            Whether to order bars from largest to smallest based on counts.
        as_percentage: bool
            Whether to plot values as percentages instead of raw counts.
        **kwargs:
            Additional keyword arguments passed to sns.barplot.
    """
    # Check if obs_column is in the AnnData object
    if obs_column not in adata.obs.columns:
        raise ValueError(f"'{obs_column}' is not in `adata.obs`.")

    # Check if hue or groupby_column are valid if provided
    if hue and hue not in adata.obs.columns:
        raise ValueError(f"'{hue}' is not in `adata.obs`.")
    if groupby_column and groupby_column not in adata.obs.columns:
        raise ValueError(f"'{groupby_column}' is not in `adata.obs`.")

    # Extract data
    obs_data = adata.obs[[obs_column]]
    if hue:
        obs_data[hue] = adata.obs[hue]
    if groupby_column:
        obs_data[groupby_column] = adata.obs[groupby_column]

    # Count occurrences
    if hue or groupby_column:
        if hue and groupby_column:
            count_data = obs_data.groupby([obs_column, hue, groupby_column]).size().reset_index(name='count')
        elif hue:
            count_data = obs_data.groupby([obs_column, hue]).size().reset_index(name='count')
        else:  # Only group by groupby_column
            count_data = obs_data.groupby([obs_column, groupby_column]).size().reset_index(name='count')

        # Convert to percentage if requested
        if as_percentage:
            if hue:
                total_counts = count_data.groupby([hue])['count'].transform('sum')
            else:
                total_counts = count_data.groupby([groupby_column])['count'].transform('sum')
            count_data['count'] = (count_data['count'] / total_counts) * 100
    else:
        # Just a simple count of the `obs_column`
        count_data = obs_data[obs_column].value_counts(normalize=as_percentage).reset_index()
        count_data.columns = [obs_column, 'count']
        if as_percentage:
            count_data['count'] *= 100

    # Order the bars if requested
    if isinstance(ordered, list):
        order=ordered
    elif ordered:
        order = count_data.groupby(obs_column)['count'].sum().sort_values(ascending=False).index.tolist()
    else:
        order = adata.obs[obs_column].cat.categories.tolist() if pd.api.types.is_categorical_dtype(
            adata.obs[obs_column]) else None

    # Determine the palette
    palette = None
    if hue:
        if f"{hue}_colors" in adata.uns:
            unique_hue_values = (
                adata.obs[hue].cat.categories
                if pd.api.types.is_categorical_dtype(adata.obs[hue])
                else pd.unique(adata.obs[hue])
            )
            palette = {str(val): color for val, color in zip(unique_hue_values, adata.uns[f"{hue}_colors"])}
    elif f"{obs_column}_colors" in adata.uns:
        unique_values = (
            adata.obs[obs_column].cat.categories
            if pd.api.types.is_categorical_dtype(adata.obs[obs_column])
            else pd.unique(adata.obs[obs_column])
        )
        palette = {str(val): color for val, color in zip(unique_values, adata.uns[f"{obs_column}_colors"])}
        if ordered:
            palette = {key: palette[key] for key in order}

    # Create the barplot
    plt.figure(figsize=figsize)
    if hue and groupby_column:
        sns.barplot(
            data=count_data,
            x=obs_column,
            y='count',
            hue=hue,
            palette=palette,
            order=order,
            ci="sd",  # Seaborn will automatically calculate the standard deviation for error bars
            **kwargs
        )
    elif hue:
        sns.barplot(
            data=count_data,
            x=obs_column,
            y='count',
            hue=hue,
            palette=palette,
            order=order,
            ci="sd",  # Seaborn will automatically calculate the standard deviation for error bars
            **kwargs
        )
    elif groupby_column:
        sns.barplot(
            data=count_data,
            x=obs_column,
            y='count',
            palette=palette,
            order=order,
            **kwargs
        )
    else:
        sns.barplot(
            data=count_data,
            x=obs_column,
            y='count',
            palette=palette,
            order=order,
            **kwargs
        )

    # Add labels and title
    plt.xlabel(obs_column)
    plt.ylabel('Percentage' if as_percentage else 'Count')
    plt.title(f"Abundance of {obs_column} values" +
              (f" grouped by {hue}" if hue else "") +
              (f" and {groupby_column}" if groupby_column else "") +
              (" (as %)" if as_percentage else ""))
    plt.xticks(rotation=45, ha='right')
    if hue:
        plt.legend(title=hue)
    plt.tight_layout()

    # Save the plot if a file path is provided
    if save:
        plt.savefig(save, bbox_inches='tight')
        print(f"Plot saved to {save}")
    plt.close()
def palantir_psuedotime_routine_external_datasets(adata,start_and_end_states,celltype_obs='Cell Subtype',pca_key='X_pca_harmony',save_prefix=None,adata_name=None):
    '''Runs Palantir pseudttime routine across Cap1 VEC and Art EC
    adata: anndata object with just Cap1, arterial, and venous EC
    start_and_end_states: used to define root and terminus for pseudotime branches dictionary defined:
    {'root_ct':('Cap1',umap_axis,'min'),
    'terminal_cts' :{'Arterial EC':(0,'max'),
     'Venous_EC':(umap_axis,'max')
     }
      }
    celltype_obs: obs column with cell type metadata
    pca_key: key for palantir to use for pseudotimes
    save_prefix: folder to save files out at
    adata_name: name to save on plots
    '''

    root_ct,umap_axis,arg = start_and_end_states['root_ct']
    terminal_cts = start_and_end_states['terminal_cts'].keys()
    palantir.utils.run_diffusion_maps(adata,n_components=5,pca_key=pca_key,)
    fig = palantir.plot.plot_diffusion_components(adata)[0]
    fig.tight_layout()
    fig.savefig(f'{save_prefix}/{adata_name}_palantir_diffusion_components.png')
    plt.close()
    palantir.utils.determine_multiscale_space(adata)
    palantir.utils.run_magic_imputation(adata)

    subset = adata[adata.obs[celltype_obs] == root_ct]
    umap_values = subset.obsm['X_umap'][:, umap_axis]
    if arg=='max':
        idx = np.argmax(umap_values)
    else:
        idx = np.argmin(umap_values)
    root_cell = subset.obs_names[idx]
    terminal_states = []
    for ct in terminal_cts:
        umap_axis,arg = start_and_end_states['terminal_cts'][ct]
        subset = adata[adata.obs[celltype_obs] == ct]
        umap_values = subset.obsm['X_umap'][:, umap_axis]
        if arg =='max':
            idx = np.argmax(umap_values)
        else:
            idx = np.argmin(umap_values)
        terminal_states.append(subset.obs_names[idx])

    terminal_states = pd.Series(index=terminal_states, data=terminal_cts, dtype='object')

    fig = palantir.plot.highlight_cells_on_umap(adata, [root_cell]+terminal_states)[0]
    fig.tight_layout()
    fig.savefig(f'{save_prefix}/{adata_name}_palantir_terminal_cells.png')
    plt.close()

    palantir.core.run_palantir(
        adata, root_cell, num_waypoints=500, terminal_states=terminal_states
    )

    fig = palantir.plot.plot_palantir_results(adata, s=3)
    fig.tight_layout()
    fig.savefig(f'{save_prefix}/{adata_name}_palantir_results.png')
    plt.close()
    iroot = adata.obs.index.get_loc(root_cell)
    adata.uns["iroot"] = iroot
    try:
        palantir.presults.select_branch_cells(adata, q=.01, eps=.01,pseudo_time_key='palantir_pseudotime')

        fig = palantir.plot.plot_branch_selection(human_adata)
        fig.tight_layout()
        fig.savefig(f'{save_prefix}/{adata_name}_palantir_branch_selection.png')
        plt.close()
    except:
        pass

    sc.tl.dpt(adata)
    sc.pl.umap(
        adata,
        color=["dpt_pseudotime", "palantir_pseudotime"],
        color_map="viridis",
        show=False,
        save=f'{adata_name}_pseudotimes.png'
    )

    # palantir.presults.compute_gene_trends(
    #     adata,
    #     expression_key="MAGIC_imputed_data",
    #     pseudo_time_key='dpt_pseudotime'
    # )
    #
    # pk = cr.kernels.PseudotimeKernel(adata, time_key="palantir_pseudotime")
    # pk.compute_transition_matrix()
    # pk.plot_projection(basis="umap", color=celltype_obs, recompute=True,legend_loc='right margin',
    #                          save=f'{save_prefix}/{adata_name}_palantir_pseudotime_stream.png')
    # plt.close()
    return

def correlate_genes_with_pseudotime(adata, layer=None, method='pearson',pseudotime='palantir_pseudotime'):
    """
    Correlates all genes with pseudotime in an AnnData object.

    Parameters:
    - adata: AnnData object with pseudotime in `adata.obs['pseudotime']`
    - layer: (Optional) Layer to use instead of adata.X (e.g., 'log1p', 'counts')
    - method: Correlation method, either 'spearman' (default) or 'pearson'

    Returns:
    - pandas DataFrame with genes as index and columns: ['correlation', 'pval']
    """
    if pseudotime not in adata.obs:
        raise ValueError("Pseudotime must be stored in adata.obs['pseudotime'].")

    # Get expression matrix
    X = adata.X if layer is None else adata.layers[layer]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X,
                         index=adata.obs_names, columns=adata.var_names)

    # Extract pseudotime
    pseudotime = adata.obs[pseudotime]

    # Run correlation
    results = []
    for gene in X.columns:
        if method == 'spearman':
            corr, pval = spearmanr(X[gene], pseudotime)
        elif method == 'pearson':
            corr, pval = X[gene].corr(pseudotime), None  # Pearson p-value not computed here
        else:
            raise ValueError("Method must be 'spearman' or 'pearson'.")
        results.append((gene, corr, pval))

    result_df = pd.DataFrame(results, columns=['gene', 'correlation', 'pval']).set_index('gene')
    return result_df.sort_values('correlation', ascending=False)

def normalize_dataframe(df):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit the scaler on the data and transform each column
    df_normalized = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    return df_normalized

def find_gene_overlap_in_pseudotimes(adata, celltype_obs='Cell Subtype', cts=['Arterial EC','Venous EC'], top_n_genes=50, pseudotime_key='palantir_pseudotime',size=15,
                                     save_prefix=None, adata_name=None):
    corr_dfs = {}
    for ct in cts:
        ct_adata = adata[adata.obs[celltype_obs] == ct]
        print(ct)
        print(ct_adata)
        df = correlate_genes_with_pseudotime(ct_adata, method='pearson', pseudotime=pseudotime_key)
        corr_dfs[ct] = df.dropna(how='all')
    ct0_large_genes = corr_dfs[cts[0]].head(top_n_genes).index.tolist()
    ct1_large_genes = corr_dfs[cts[1]].head(top_n_genes).index.tolist()
    ct0_small_genes = corr_dfs[cts[0]].tail(top_n_genes).index.tolist()[::-1]
    ct1_small_genes = corr_dfs[cts[1]].tail(top_n_genes).index.tolist()[::-1]

    # Create the Venn diagram
    venn = venn2([set(ct0_large_genes), set(ct1_large_genes)],
                 set_labels=(cts[0], cts[1]),
                 set_colors=('#4A90E2', '#E35D6A'),
                 alpha=0.7)

    # Optional: Customize font size
    for text in venn.set_labels:
        text.set_fontsize(12)
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(12)

    # Show the plot
    plt.title("Top 50 genes positively correlated with pseudotime")
    plt.savefig(f'{save_prefix}/{adata_name}venn_diagram_large.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Create the Venn diagram
    venn = venn2([set(ct0_small_genes), set(ct1_small_genes)],
                 set_labels=(cts[0], cts[1]),
                 set_colors=('#4A90E2', '#E35D6A'),
                 alpha=0.7)

    # Optional: Customize font size
    for text in venn.set_labels:
        text.set_fontsize(12)
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(12)

    # Show the plot
    plt.title("Top 50 genes positively correlated with pseudotime")
    plt.savefig(f'{save_prefix}/{adata_name}venn_diagram_small.png', dpi=300, bbox_inches='tight')
    plt.close()


    large_genes = [x for x in ct0_large_genes if x in ct1_large_genes]
    small_genes = [x for x in ct0_small_genes if x in ct1_small_genes]
    sc.tl.score_genes(adata, large_genes, score_name='large_score')
    sc.tl.score_genes(adata, small_genes, score_name='small_score')
    adata.obs['Vessel size score'] = adata.obs['large_score'] - adata.obs['small_score']
    scaler = MinMaxScaler(feature_range=(0, 1))
    adata.obs['Vessel size score'] = scaler.fit_transform(adata.obs[['Vessel size score']])
    adata.obs['Vessel size category'] = pd.cut(adata.obs['Vessel size score'], bins=4,
                                               labels=['capillary', 'small', 'medium', 'large'])
    sc.pl.umap(adata, color=['Vessel size score'], cmap='Oranges', size=size, frameon=False,
               save=f'_{adata_name}_vessel_size_score.png')
    sc.pl.umap(adata, color=['Vessel size category'], cmap='viridis', size=size, frameon=False,
               save=f'_{adata_name}_vessel_size_category.png')
    sc.pl.umap(adata, color=large_genes + small_genes, cmap='viridis', hspace=0.5, save=f'_{adata_name}_allsize.png')
    sc.pl.umap(adata, color=['Cell Subtype'], size=size, frameon=False,
               save=f'_{adata_name}_cell_subtype.png')

    adata.write(f'{save_prefix}/vessel_size.gz.h5ad', compression='gzip')
    return

def custom_dotplot(adata, genes, x_obs, y_obs, x_order=None, y_order=None,min_expr=0.1, cmap='RdBu_r',dot_max_size=300, pad=0.5,scale_by_gene=False, figsize=None,save=None, dpi=300,show_gridlines=False):
    """
    Custom dotplot that mimics scanpy's style but allows:
    - Custom x/y groupings
    - Split x-axis for condition + gene
    - Color = average expression, Size = percent expressing
    - Optional scaling and exporting
    """
    adata.obs[x_obs] = adata.obs[x_obs].astype('category')
    adata.obs[y_obs] = adata.obs[y_obs].astype('category')

    if x_order is None:
        x_order = adata.obs[x_obs].cat.categories.tolist()
    if y_order is None:
        y_order = adata.obs[y_obs].cat.categories.tolist()

    df = sc.get.obs_df(adata, keys=[x_obs, y_obs] + genes, layer=None)

    results = []
    for gene in genes:
        for x_val in x_order:
            for y_val in y_order:
                group = df[(df[x_obs] == x_val) & (df[y_obs] == y_val)]
                if group.shape[0] == 0:
                    continue
                expr = group[gene]
                avg_expr = expr.mean()
                prop_expr = (expr > min_expr).mean()
                results.append({
                    "gene": gene,
                    "x_group": x_val,
                    "y_group": y_val,
                    "avg_expr": avg_expr,
                    "prop_expr": prop_expr
                })

    plot_df = pd.DataFrame(results)
    plot_df["x_label"] = plot_df["gene"] + "\n" + plot_df["x_group"]

    # Scale avg_expr within gene
    if scale_by_gene:
        plot_df["scaled_expr"] = plot_df.groupby("gene")["avg_expr"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        color_col = "scaled_expr"
    else:
        color_col = "avg_expr"

    # X/Y axis label arrangement
    x_labels = []
    x_groups = []
    genes_list = []
    for gene in genes:
        for x_val in x_order:
            x_labels.append(f"{x_val}\n{gene}")
            x_groups.append(x_val)
            genes_list.append(gene)

    y_labels = y_order

    if figsize is None:
        figsize = (len(x_labels) * 0.6 + 2, len(y_labels) * 0.6 + 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    for _, row in plot_df.iterrows():
        x = x_labels.index(f"{row['x_group']}\n{row['gene']}")
        y = y_labels.index(row["y_group"])
        ax.scatter(
            x, y,
            s=row["prop_expr"] * dot_max_size,
            c=[row[color_col]],
            cmap=cmap,
            vmin=plot_df[color_col].min(),
            vmax=plot_df[color_col].max(),
            edgecolor='black',
            linewidth=0.5
        )

    # Optional gridlines
    if show_gridlines:
        for y in range(len(y_labels)):
            ax.axhline(y, color='lightgray', linestyle=':', linewidth=0.5)

    # Vertical dashed lines between gene groups
    for i in range(1, len(genes)):
        xpos = i * len(x_order) - 0.5
        ax.axvline(x=xpos, color='gray', linestyle='--', linewidth=1)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_groups, rotation=0, ha='center')
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlim(-pad, len(x_labels) - 1 + pad)
    ax.set_ylim(-pad, len(y_labels) - 1 + pad)
    ax.invert_yaxis()
    ax.set_xlabel('')
    ax.set_ylabel(y_obs)

    # # Add gene names on second x-axis
    ax_gene = ax.secondary_xaxis('top')
    gene_locs = range(len(x_labels))
    gene_locs = [(gene_locs[i] + gene_locs[i + 1]) / 2 for i in range(len(gene_locs) - 1)][::2]
    ax_gene.set_xticks(gene_locs)
    # if len(set(genes_list)) == 1:
    #     ax_gene.set_xticklabels(genes_list[::2], rotation=0, ha='center')
    # else:
    ax_gene.set_xticklabels(genes_list[::2], rotation=0, ha='center')
    # # ax_gene.set_xlabel("Gene")

    # Colorbar
    norm = plt.Normalize(plot_df[color_col].min(), plot_df[color_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02)
    cbar.set_label('Mean expression\nin group')

    # Dot size legend
    prop_vals = plot_df["prop_expr"]
    # min_pct = int(np.floor(prop_vals.min() * 100 / 5) * 5)
    # max_pct = int(np.ceil(prop_vals.max() * 100 / 5) * 5)
    max_pct=100
    min_pct=0
    possible_labels = [i for i in range(min_pct, max_pct + 1) if i % 5 == 0]
    num_labels = min(4, len(possible_labels))
    legend_labels = np.linspace(min_pct, max_pct, num_labels, dtype=int)

    handles = [
        plt.scatter([], [], s=(pct / 100) * dot_max_size, c='gray',
                    edgecolors='black', linewidth=0.5, label=f"{pct}%")
        for pct in legend_labels
    ]

    # Move axis to left to make room for both legends
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])

        # --- Dot size legend with clean rounding ---
    min_pct_raw = 25#plot_df["prop_expr"].min() * 100
    max_pct_raw = 100#plot_df["prop_expr"].max() * 100

    min_pct = int(np.floor(min_pct_raw / 5) * 5)
    max_pct = int(np.ceil(max_pct_raw / 5) * 5)

    # Generate 4 nicely rounded ticks between min and max
    if max_pct - min_pct < 15:
        ticks = np.linspace(min_pct, max_pct, 4)
    else:
        ticks = np.round(np.linspace(min_pct, max_pct, 4))

    ticks = np.clip(ticks, 0, 100).astype(int)

    handles = [
        plt.scatter([], [], s=(p / 100) * dot_max_size, c='gray', edgecolors='black')
        for p in ticks
    ]
    labels = [f"{p}%" for p in ticks]

    fig.legend(
        handles,
        labels,
        title="Pct. Expressing",
        loc='center right',
        bbox_to_anchor=(1+(1/figsize[0]), 0.6),
        frameon=True
    )

    fig.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save}")
    else:
        plt.show()