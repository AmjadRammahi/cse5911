import math
import logging
from pprint import pprint

from numpy.core.records import array
import src.global_var
from src.settings import Settings
from src.izgbs import izgbs
import numpy as np

def evaluate_location(location_data: array) -> array:
    '''
        Runs IZGBS on a specified location, return results in numpy array best_results.

        Params:
            location_data (numpy array) : location data, service_req.

        Returns:
            (numpy array) : contains all field that going to fill in result_df.
    '''
    best_result = []
    start_val = math.ceil((src.global_var.MAX_MACHINES - 1) / 2)
    sas_alpha_value = Settings.ALPHA_VALUE / math.log2(src.global_var.MAX_MACHINES - 1)
    loc_res = izgbs(
        src.global_var.MAX_MACHINES,
        start_val,
        Settings.MIN_ALLOC,
        sas_alpha_value,
        location_data[0],
        location_data[1]
    )
    loc_feas = loc_res[loc_res[:,1] == 1]
    if loc_feas.size != 0:
        loc_feas_min = loc_feas[0]
        # populate overall results with info for this location
        best_result = np.delete(loc_feas_min, 1)
        
    return best_result