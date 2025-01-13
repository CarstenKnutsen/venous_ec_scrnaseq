import scanpy as sc
import pandas as pd
import numpy as np
import os
import json
import shutil
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns

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
    if ordered:
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
