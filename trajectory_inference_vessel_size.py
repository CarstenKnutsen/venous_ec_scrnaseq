'''Goal:Trajectory inference for vessel size in endos and mural cells
Date:250205
Author: Carsten Knutsen
conda_env:trajectory_inference
'''


import os
import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
import palantir
import cellrank as cr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data = 'data/single_cell_files/scanpy_files'
figures = 'data/figures/trajectory_inference'
adata_name = "venous_ec"
os.makedirs(figures, exist_ok=True)
sc.set_figure_params(dpi=300, format="png")
sc.settings.figdir = figures
scv.settings.figdir = figures
sc.settings.autoshow = False
scv.settings.autoshow = False
comps = {
    'vascular_endothelial':{'cts':['Arterial EC', 'Cap1','Venous EC'],
                 'root_ct':'Cap1',
                 'terminal_cts':['Arterial EC','Venous EC']
                 },
    'Mural':{'cts':['Pericyte', 'Vascular smooth muscle'],
                 'root_ct':'Cap1',
                 'terminal_cts':['Cap2']
                 },

}
celltype = 'Cell Subtype_no_cc'
def run_velocity_routine(adata,adata_v,celltype,root_ct,terminal_cts,figure_dir,output_file,umap=False):
    ## Lots of requirements here
    sc.settings.figdir = figure_dir
    scv.settings.figdir = figure_dir
    sc.pp.highly_variable_genes(adata, batch_key='Library')
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X_pca")
    if umap:
        sc.tl.umap(adata,min_dist=0.5)
        sc.pl.umap(adata,color=['Library','Treatment',celltype],save='_metadata')
    adata_v_trim = adata_v[adata.obs_names, :].copy()
    adata_v_trim = scv.utils.merge(adata, adata_v_trim)
    adata_v_trim.X = adata_v_trim.layers['raw'].copy()
    scv.pp.filter_and_normalize(adata_v_trim, enforce=True)
    scv.pp.moments(adata_v_trim, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata_v_trim)
    for mode in [
        # 'stochastic',
        # 'steady_state', crashing running all three see github issue
        'dynamical'
    ]:
        scv.tl.velocity(adata_v_trim, mode=mode)
        scv.tl.velocity_graph(adata_v_trim)
        scv.pl.velocity_embedding_stream(adata_v_trim, basis='umap', density=3, color=celltype,legend_loc='right margin',
                                         title='', show=False, save=f'{mode}_velocity.png')
    scv.tl.latent_time(adata_v_trim)
    scv.tl.terminal_states(adata_v_trim)
    scv.tl.velocity_pseudotime(adata_v_trim)
    scv.tl.paga(adata_v_trim, groups=celltype)
    scv.pl.scatter(adata_v_trim, color=["root_cells", "end_points"],
                   save='_velocity_states.png')
    scv.pl.paga(adata_v_trim, groups=celltype,save='_celltype_paths.png')
    adata_v_trim.X = adata_v_trim.layers['log1p'].copy()
    palantir.utils.run_diffusion_maps(adata_v_trim,
                                               n_components=10)
    fig = palantir.plot.plot_diffusion_components(adata_v_trim)[0]
    fig.tight_layout()
    fig.savefig(f'{figure_dir}/palantir_diffusion_components.png')
    plt.close()
    palantir.utils.determine_multiscale_space(adata_v_trim)

    palantir.utils.run_magic_imputation(adata_v_trim)


    root_cell = palantir.utils.early_cell(adata_v_trim, root_ct, celltype,fallback_seed=0)
    terminal_states = palantir.utils.find_terminal_states(adata_v_trim,
                                                          terminal_cts,
                                                          celltype)

    fig = palantir.plot.highlight_cells_on_umap(adata_v_trim, terminal_states)[0]
    fig.tight_layout()
    fig.savefig(f'{figure_dir}/palantir_terminal_cells.png')
    plt.close()

    palantir.core.run_palantir(
        adata_v_trim, root_cell, num_waypoints=500, terminal_states=terminal_states
    )

    fig = palantir.plot.plot_palantir_results(adata_v_trim, s=3)
    fig.tight_layout()
    fig.savefig(f'{figure_dir}/palantir_results.png')
    plt.close()

    try:
        palantir.presults.select_branch_cells(adata_v_trim, q=.01, eps=.01)

        fig = palantir.plot.plot_branch_selection(adata_v_trim)
        fig.tight_layout()
        fig.savefig(f'{figure_dir}/palantir_branch_selection.png')
        plt.close()

        palantir.presults.compute_gene_trends(
            adata_v_trim,
            expression_key="MAGIC_imputed_data",
        )
    except:
        pass

    sc.tl.diffmap(adata_v_trim)
    iroot = adata_v_trim.obs.index.get_loc(root_cell)
    scv.pl.scatter(
        adata_v_trim,
        basis="diffmap",
        c=[celltype, iroot],
        legend_loc="right",
        components=["2, 3"],
        show=False,
        save=f'diffmap_{celltype}_root_cell.png'
    )

    adata_v_trim.uns["iroot"] = iroot

    sc.tl.dpt(adata_v_trim)
    sc.pl.embedding(
        adata_v_trim,
        basis="umap",
        color=["dpt_pseudotime", "palantir_pseudotime",'velocity_pseudotime','latent_time'],
        color_map="viridis",
        show=False,
        save='_pseudotimes.png'
    )

    pk = cr.kernels.PseudotimeKernel(adata_v_trim, time_key="palantir_pseudotime")
    pk.compute_transition_matrix()
    pk.plot_projection(basis="umap", color=celltype, recompute=True,legend_loc='right margin',
                             save=f'{figure_dir}/palantir_pseudotime_stream.png')
    adata_v_trim.write(output_file, compression='gzip')
    return

color_dict ={'Arterial EC': '#ffff00',
 'Cap1': '#1ce6ff',
 'Cap1_Cap2': '#ff34ff',
 'Cap2': '#ff4a46',
 'Lymphatic EC': '#008941',
 'Proliferating Cap': '#006fa6',
 'Proliferating Venous EC': '#a30059',
 'Systemic Venous EC': '#ffdbe5',
 'Venous EC': '#7a4900'}

if __name__ == '__main__':
    adata = sc.read(f'{data}/{adata_name}_celltyped_no_cc.gz.h5ad')
    adata_velocity = sc.read(f'{data}/{adata_name}_all_cells_velocyto.gz.h5ad')
    for comp,comp_dict in comps.items():
        figures_comp = f'{figures}/{comp}'
        os.makedirs(figures_comp, exist_ok=True)
        print(comp)
        cts = comp_dict['cts']
        root_ct = comp_dict['root_ct']
        terminal_cts = comp_dict['terminal_cts']
        adata_cts = adata[adata.obs[celltype].isin(cts)].copy()
        adata_cts.X = adata_cts.layers['log1p_cc_regress'].copy()
        # sc.pp.regress_out(adata_cts, ['S_score', 'G2M_score'])
        new_colors = []
        for ct in adata_cts.obs[celltype].cat.categories:
            if ct == 'Proliferating EC':
                ct = 'Proliferating Cap'
            new_colors.append(color_dict[ct])
        adata_cts.uns[f'{celltype}_colors'] = new_color
        run_velocity_routine(adata_cts,adata_velocity,celltype,root_ct,terminal_cts,figures_comp,f'{data}/{adata_name}_{comp}_velocity.gz.h5ad',umap=True)
