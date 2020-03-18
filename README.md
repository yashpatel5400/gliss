# GLISS

The repo includes the code and notebooks of GLISS, a strategy to analyze spatially-varying genes by integrating two data sources: (1) Spatial Gene Expression (SGE) data such as image-based fluorescence in situ hybridization techniques, and (2) dissociated whole-transcriptome single-cell RNA-sequencing (scRNA-seq) data. GLISS utilizes a graph-based association measure that we developed to select and link genes that are spatially-dependent in both data sources. GLISS can discover new spatial genes and recover cell locations in scRNA-seq data from landmark genes determined from SGE data. GLISS also offers a new dimension reduction technique to cluster the genes, while accounting for the inferred spatial structure of the cells. We demonstrate the utility of GLISS on simulated and real data sets, including two SGE data sets on the mouse olfactory bulb and breast cancer biopsies, and two integrative spatial studies of the mammalian liver and intestine.

## Installation

    pip3 install -r requirements.txt

## Quick Tutorial


    from plot_utils import plot_ground_truth_heatmap

## Simulation Setup and Results

### Simulated SGE Analysis

Core functions:

    from sim_utils import multi_2D_pattern, add_gaussian_noise
    from main_methods import select_spatial_genes
    from sim_utils import complex_eval_spatial_sim, read_trial_data

Notebooks:
- `gliss/nb/sim_sge_setup.ipynb`
- `gliss/nb/sim_sge_run_gliss.ipynb`
- `gliss/nb/sim_sge_analysis.ipynb`

#### Setup SGE simulation

Simulated data generation:
`gliss/nb/sim_sge_setup.ipynb`

    from sim_utils import multi_2D_pattern, add_gaussian_noise
    from io_utils import save_data_to_file

Required raw data paths:
`/share/PI/sabatti/sc_data/spatial_ge/scGCO/`

Output data path:

`/Users/jjzhu/Google Drive/_GLISS/data/space_comp_sim/20191104`(local)

`/share/PI/sabatti/feat_viz/space_comp_sim/20191104` (sherlock)
- `sim_setup.csv`: simulation parameters
- `locs_2d.csv`: 2-D coordinates
- `sim_data/mtx_*_*.csv`: gene expression matrices


#### Run SGE methods

Run gliss on the simulated data:
`gliss/nb/sim_sge_run_gliss.ipynb`

    from main_methods import select_spatial_genes
    from sim_utils import complex_eval_spatial_sim, read_trial_data

Running other benchmarks:
- Dependencies: `pip3 install -r benchmarks/requirements_sge_methods.txt`
- Virtual environment name: `space_met`
- Example script: `gliss/benchmarks/sge_methods/launch_scgco.sh`
- SpatialDE: `gliss/benchmarks/sge_methods/run_spatialde.py`
- scGCO: `gliss/benchmarks/sge_methods/run_scgco.py`

#### Analyze SGE outputs
Analyze the FDR and power of SV selection:
`gliss/nb/sim_sge_analysis.ipynb`


### Simulated scRNA-seq Analysis

Core functions:

    from main_methods import run_procedure
    from spline_utils import setup_basis, spline_fit
    from main_methods import compute_all_embeddings

Notebooks:
- `gliss/nb/sim_scrna_setup_and_run_gliss.ipynb`
- `gliss/benchmarks/ti_methods/sim_scrna_Palantir_PAGA.ipynb`
- `gliss/benchmarks/ti_methods/sim_scrna_Palantir_PAGA_LM_SV.ipynb`
- `gliss/nb/sim_scrna_analysis_comparison.ipynb`

#### Setup scRNA-seq simulation and run GLISS

Simulated data generation:
`gliss/nb/sim_scrna_setup_and_run_gliss.ipynb`

    from io_utils import save_data_to_file
    from sim_utils import get_corr_sim  # get simulation parameters
    from sim_utils generate_invariant_data, model_corr_noise

- `regime_0`: $\rho = 0.01$
- `regime_1`: $\rho = 0.2$
- `regime_2`: $\rho = 0.4$

Required raw data paths:

`/share/PI/sabatti/feat_viz/corr_sim/regime_*/` (sherlock)

`/Users/jjzhu/Google Drive/_GLISS/data/scrna_sim` (local)

- `data_dict.pkl`: the matrix data (z: LM mtx, x: non-LM, lam: order)
- `var_df.csv`: information of each feature

Running GLISS:

    from main_methods import run_procedure
    from spline_utils import setup_basis, spline_fit
    from main_methods import compute_all_embeddings

Output data path:
`/share/PI/sabatti/feat_viz/corr_sim/regime_*/`
- `method_result.pkl`: SV selection result and spatial inference from GLISS
    - FILL
    - FILL
- `coeff_matrix.npy`: coefficient matrix after spline fitting
- `embed_dict.pkl`: visualizations of various gene representations
    - FILL
    - FILL

#### Running other benchmarks:
- Dependencies: `pip3 install -r benchmarks/requirements_ti_methods.txt`
- Virtual environment name: `ti`
- Vanilla PAGA/Palantir: `gliss/benchmarks/ti_methods/sim_scrna_Palantir_PAGA.ipynb`
- LM+SV PAGA/Palantir: `gliss/benchmarks/ti_methods/sim_scrna_Palantir_PAGA_LM_SV.ipynb`

#### Analyze scRNA-seq outputs

    from sim_utils import get_corr_sim, generate_invariant_data, model_corr_noise
    from spline_utils import setup_basis # for reconstruction from coefficients

`gliss/nb/sim_scrna_analysis_comparison.ipynb`


## Real Data Analysis


### Hepatotype Dataset

Data paths:
- `.data/liver2k/analysis_on_data_original/data`
- `.data/analysis_050719` (GLISS results)
- `.data/analysis_060719/hepa_data/ti_methods`

Notebooks:

- `gliss/nb/real_hepa_setup_part0_preprocess.ipynb`
- `gliss/nb/real_hepa_setup_part1_latentspace.ipynb`
- `gliss/nb/real_hepa_setup_part2_clustering.ipynb`
- `gliss/benchmarks/ti_methods/real_hepa_run_paga.ipynb`
- `gliss/benchmarks/ti_methods/real_hepa_run_palantir.ipynb`
- `gliss/nb/real_hepa_analysis_part1.ipynb`

#### Part 0: preprocessing

`gliss/nb/real_hepa_setup_part0_preprocess.ipynb`

- load processed data directly with: `liver_info.load_processsed_hepatocyte_data`:
    - `data/obs_info.csv`: meta information for the observations
    - `data/var_info.csv`: meta information for the genes
    - `data/matrix_unscaled.csv`: the expression matrix with 8883 genes

- load the zonations from the original paper with `liver_info.load_zonation_result`
    - downloaded the table `table_s2_reform.csv` from https://www.nature.com/articles/nature21065
    - the function will take the directory as input and compute maximum probabilities

Processed data is saved at: `/share/PI/sabatti/sc_data/liver2k/analysis_on_data_original/data`

#### Part 1: SV gene selection + latent space infernce


`gliss/nb/real_hepa_setup_part1_latentspace.ipynb`

`gliss/nb/real_hepa_analysis_part1.ipynb`

    from liver_info import load_processsed_hepatocyte_data, load_zonation_result, get_known_liver_markers
    from io_utils import load_all_pipeline_results

Output data path:
`/share/PI/sabatti/feat_viz/real_analysis_result/analysis_050719`

- There are multiple latent space inferncee options, but only `graph_vanilla` is used.
- i.e.,  `/share/PI/sabatti/feat_viz/real_analysis_result/analysis_050719/result_graph_vanilla`

Other methods:
- PAGA:  `gliss/benchmarks/ti_methods/real_hepa_run_paga.ipynb`
- Palantir:  `gliss/benchmarks/ti_methods/real_hepa_run_palantir.ipynb`
- Output: `/share/PI/sabatti/feat_viz/real_analysis_result/analysis_060719/hepa_data/ti_methods/*_obs_df.csv`

#### Part 2: gene clustering

`gliss/nb/real_hepa_setup_part2_clustering.ipynb`

Temporary outputs:
`/scratch/PI/sabatti/spatial_subplots/hepa_gene_clusters_ours_k_13/`

### Enterocyte Dataset

Data paths:
- `./data/intestine2k`
- `./analysis_060719/entero_data`
- `./analysis_060719/entero_data/ti_methods`

Notebooks:


#### Part 0: preprocessing

`gliss/nb/real_entero_setup_part0_preprocess.ipynb`


#### Part 1: SV gene selection + latent space infernce

`gliss/nb/real_entero_setup_part1_latentspace.ipynb`

`gliss/nb/real_entero_analysis_part1.ipynb`

    from intestine_info import load_processed_enterocyte_data, load_original_entero_zonation
    from intestine_info import get_intestine_rna_lm_genes
    from io_utils import load_all_pipeline_results, load_data_from_file

Output data path:
`/share/PI/sabatti/feat_viz/real_analysis_result/analysis_060719/entero_data/results_our_lm/`

- There are multiple latent space inferncee options, but only `graph_vanilla` is used.
- i.e.,  `/share/PI/sabatti/feat_viz/real_analysis_result/analysis_060719/entero_data/results_our_lm/result_graph_vanilla`

Other methods:
- PAGA:  `gliss/benchmarks/ti_methods/real_entero_run_paga.ipynb`
- Palantir:  `gliss/benchmarks/ti_methods/real_entero_run_palantir.ipynb`
- Output: `/share/PI/sabatti/feat_viz/real_analysis_result/analysis_060719/entero_data/ti_methods/*_obs_df.csv`

#### Part 2: gene clustering

`gliss/nb/real_entero_setup_part2_clustering.ipynb`

`gliss/nb/real_entero_analysis_part2.ipynb`

Temporary outputs:
`/scratch/PI/sabatti/spatial_subplots/intestine_gene_clusters_ours_k_15/`
`/scratch/PI/sabatti/spatial_subplots/intestine_gene_clusters_ours_k_9/`


### Real SGE Data Sets


Data paths:

`/share/PI/sabatti/sc_data/spatial_ge/scGCO` (sherlock)

`/Users/jjzhu/Google Drive/_GLISS/data/scGCO` (local)

- `./data/Raw_data/MOB-breast-cancer/`
- `./data/BreastCancer/*/Layer*_result_df.csv`
- `./data/MouseOB/*/Rep*_result_df.csv`

Notebooks:
- `gliss/nb/real_sge_setup_bc.ipynb`
- `gliss/nb/real_sge_setup_mob.ipynb`
- `gliss/nb/real_sge_analysis.ipynb`

#### Run GLISS

Notebooks:
- `gliss/nb/real_sge_setup_bc.ipynb`
- `gliss/nb/real_sge_setup_mob.ipynb`

    from main_methods import select_spatial_genes
    from general_utils import read_spatial_expression, normalize_count_cellranger
    from io_utils import save_data_to_file, load_data_from_file


#### Analyze and compare methods

`gliss/nb/real_sge_analysis.ipynb`

Our data stored as `csv` files in:
- `/share/PI/sabatti/sc_data/spatial_ge/scGCO/data/BreastCancer/our_results/`
- `/share/PI/sabatti/sc_data/spatial_ge/scGCO/data/MouseOB/our_results/`

Other method results are stored in:
- `/share/PI/sabatti/sc_data/spatial_ge/scGCO/data/BreastCancer/scGCO_results/Layer*_result_df.csv`
- `/share/PI/sabatti/sc_data/spatial_ge/scGCO/data/BreastCancer/spatialDE_results/Layer*_result_df.csv`
- `/share/PI/sabatti/sc_data/spatial_ge/scGCO/data/MouseOB/scGCO_results/Rep*_result_df.csv`
- `/share/PI/sabatti/sc_data/spatial_ge/scGCO/data/MouseOB/spatialDE_results/Rep*_result_df.csv`

------------------


## Old Simulation Instructions

#### Step-by-step guide:

1. Update info in `get_sim_id_classes()` in `sim_utils.py`, two classes are available now:
    - `full_sim_ids`: full-scale simulation
    - `strict_sim_ids`: simulation with strict order recovery
2. Change parameters in `get_sim_params()` in `sim_utils.py`
    - this is a giant config chuck in dictionary; make sure to add the right if statement
3.  Run `generate_regime_info(sim_id)` and check params with quick visualizations:
    - `check_spike_groups(sim_id)`: visualize the non-null templates
    - `check_example_noise_matrix(sim_id)`: visualize the matrix that includes null and non-nulls
4. Test the simulation by running:
    - `launch_regime_feat_sel(sim_id, 0, test=True)`: full-scale simulation
    - `launch_strict_order_pipeline(sim_id, 0, test=True)`: simulation with strict order recovery
5. Create slurm jobs:
    - mote the estimated time and make adjustment to `create_trial_jobs(sim_id)` if necessary
    - alternatively, you can test with `bash */slurm_job.sh` in interactive mode


#### Setup in notebooks:
- `notebooks/setup_main_sim_1D.ipynb`
- `notebooks/setup_supp_sim_strict_1D.ipynb`

#### View results in notebook:
- `notebooks/analysis_main_sim_1D.ipynb`
- `notebooks/analysis_supp_sim_strict_1D.ipynb`
