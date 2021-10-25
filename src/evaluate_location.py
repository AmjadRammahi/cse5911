import math
import logging
from numba import jit
from src.settings import Settings
from src.izgbs import izgbs
import numpy as np


def evaluate_location(location_data: dict) -> dict:

    '''
        Runs IZGBS on a specified location, return results in dict best_results.

        Params:
            location_data (dict) : location data.

        Returns:
            (dict) : contains all field that going to fill in result_df.
    '''
    best_result = {}

    start_val = math.ceil((location_data['NUM_MACHINES'] - 1) / 2)
    sas_alpha_value = Settings.ALPHA_VALUE / math.log2(location_data['NUM_MACHINES'] - 1)

    loc_res = izgbs(
        location_data['NUM_MACHINES'],
        start_val,
        Settings.MIN_ALLOC,
        sas_alpha_value,
        location_data
    )


    loc_feas = loc_res[loc_res[:,1] == 1]
    
    
    if loc_feas.size != 0:
        # find fewest feasible machines
        mach_min = loc_feas[:,0][0]
        
        # keep the feasible setup with the fewest number of machines
        loc_feas_min = loc_feas[loc_feas[:,0] == mach_min]
       
        # populate overall results with info for this location
        avg_wait = loc_feas_min[0,2]
        max_wait = loc_feas_min[0,3]
        best_result['Resource'] = mach_min
        best_result['Exp. Avg. Wait Time'] = avg_wait
        best_result['Exp. Max. Wait Time'] = max_wait
    
    
    else:

        # no feasible setups, find lowest wait time (should work out to be max machines allowed)
        max_avg = []
        max_avg.append(loc_res[:,3])
        wait_time_min = min(max_avg)
        loc_res_min = loc_res[loc_res[:,3] == wait_time_min]
        best_result['Resource'] = loc_res_min[0][0]
        best_result['Exp. Avg. Wait Time'] = loc_res_min[0,2]
        best_result['Exp. Max. Wait Time'] = loc_res_min[0,3]
    
    return best_result