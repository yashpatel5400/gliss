import os
import sys
import json
import numpy as np
from sim_utils import run_batch_dist_corr_perm 

tmp_dir = sys.argv[1]
i_batch = int(sys.argv[2])

with open(os.path.join(tmp_dir, 'batch_job_params.json'), 'r') as infile:
    params = json.load(infile)
params["i_batch"] = i_batch

run_batch_dist_corr_perm(params)
