# GLISS

## Installation

    pip3 install -r requirements.txt

## Quick Tutorial 

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

Required raw data paths:
`/share/PI/sabatti/sc_data/spatial_ge/scGCO/`

Simulated data generation:
`gliss/nb/sim_sge_setup.ipynb`

    from sim_utils import multi_2D_pattern, add_gaussian_noise
    from io_utils import save_data_to_file

Output data path:
`/share/PI/sabatti/feat_viz/space_comp_sim/20191104`
- `sim_setup.csv`: simulation parameters
- `locs_2d.csv`: 2-D coordinates
- `sim_data/mtx_*_*.csv`: gene expression matrices 


#### Run SGE methods

Run gliss on the simulated data:
`gliss/nb/sim_sge_run_gliss.ipynb`

    from main_methods import select_spatial_genes
    from sim_utils import complex_eval_spatial_sim, read_trial_data

Running other benchmarks:
- Dependencies: `gliss/benchmarks/requirements_sge_methods.txt`
- Example script: `aloe/scripts/launch_scgco.sh` (also loads environemnt)
- SpatialDE: `gliss/scripts/run_spatialde.py`
- scGCO: `aloe/scripts/run_scgco.py`

#### Analyze SGE outputs
Run analyze the FDR and power of SV selection:
`gliss/nb/sim_sge_analysis.ipynb`

## scRNA-seq Data 


## Real Data Analysis Instructions

#### Preprocessing

- see `notebooks/setup_real_data_preprocess_hepa.ipynb`
- load processed data directly with: `liver_info.load_processsed_hepatocyte_data`:
    - `data/obs_info.csv`: meta information for the observations
    - `data/var_info.csv`: meta information for the genes
    - `data/matrix_unscaled.csv`: the expression matrix with 8883 genes
    - data available by: `wget http://stanford.edu/~jjzhu/fileshare/aloe/hepa_data.tar.gz .`
    
- load the zonations from the original paper with `liver_info.load_zonation_result`
    - download the table `table_s2_reform.csv` from https://www.nature.com/articles/nature21065
    - the function will take the directory as input and compute maximum probabilities
    
#### Pre-analysis

- see `notebooks/setup_real_data_hepa_1D.ipynb`

#### Visualize results

- see `notebooks/analysis_real_data_hepa_1D.ipynb`


## Simulation Instructions

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
