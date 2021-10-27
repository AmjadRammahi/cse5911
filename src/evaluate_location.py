import math
import logging
from pprint import pprint

import src.global_var
from src.settings import Settings
from src.izgbs import izgbs
import numpy as np



def evaluate_location(location_data: dict) -> dict:

    '''
        Runs IZGBS on a specified location, return results in dict best_results.

        Params:
            location_data (dict) : location data, service_req.

        Returns:
            (dict) : contains all field that going to fill in result_df.
    '''
    best_result = {}

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
        mach_min = loc_feas[:,0][0]
        loc_feas_min = loc_feas[loc_feas[:,0] == mach_min]
        print(loc_feas_min)
        # populate overall results with info for this location

        avg_wait = loc_feas_min[0,2]
        max_wait = loc_feas_min[0,3]
        best_result['Resource'] = mach_min
        best_result['Exp. Avg. Wait Time'] = avg_wait
        best_result['Exp. Max. Wait Time'] = max_wait

    else:
        print(loc_res[loc_res[:,3]])
        print(np.min(loc_res[loc_res[:,3]]))
        loc_res_min = np.min(loc_res[loc_res[:,3]])
        best_result['Resource'] = loc_res_min[0]
        best_result['Exp. Avg. Wait Time'] = loc_res_min[0,2]
        best_result['Exp. Max. Wait Time'] = loc_res_min[0,3]

    return best_result