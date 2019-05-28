import os
import sys
import json
import numpy as np
from sim_utils import launch_regime_feat_sel, launch_strict_order_pipeline, get_sim_id_classes
import logging

logger = logging.getLogger("feat_viz")
logging.basicConfig(format='[%(name)s %(levelname)s] %(message)s', level=logging.INFO)
logger.info("Params: {}".format(sys.argv[1:4]))


sim_id = sys.argv[1]
i_regime = int(sys.argv[2])

full_sim_ids, strict_sim_ids = get_sim_id_classes()
all_sim_ids = full_sim_ids + strict_sim_ids
assert sim_id in all_sim_ids, "{} not specified".format(sim_id)

if sim_id in full_sim_ids:
    launch_regime_feat_sel(sim_id, i_regime)
    
if sim_id in strict_sim_ids:
    launch_strict_order_pipeline(sim_id, i_regime)
