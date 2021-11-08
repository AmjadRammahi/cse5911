import math
from pprint import pprint

<<<<<<< HEAD
from numpy.core.records import array
import src.global_var
=======
>>>>>>> main
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
<<<<<<< HEAD
    best_result = []
    start_val = math.ceil((src.global_var.MAX_MACHINES - 1) / 2)
    sas_alpha_value = Settings.ALPHA_VALUE / math.log2(src.global_var.MAX_MACHINES - 1)
=======
    best_result = {}

    start_val = math.ceil((Settings.MAX_MACHINES - 1) / 2)
    sas_alpha_value = Settings.ALPHA_VALUE / math.log2(Settings.MAX_MACHINES - 1)

>>>>>>> main
    loc_res = izgbs(
        Settings.MAX_MACHINES,
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
<<<<<<< HEAD
        best_result = np.delete(loc_feas_min, 1)
        
    return best_result
=======

        best_result['Resource'] = mach_min
        best_result['Exp. Avg. Wait Time'] = loc_feas_min['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_feas_min['BatchMaxAvg']

    else:
        # no feasible setups, find lowest wait time (should work out to be max machines allowed)
        max_avg = []
        min_index = 0
        for key, result in loc_res.items():
            max_avg.append(result['BatchMaxAvg'])
            min_index = key

        loc_res_min = loc_res[min_index]

        # for key, result in loc_res.items():
        #     if result['BatchMaxAvg'] == wait_time_min:
        #         loc_res_min = result
        #         break

        best_result['Resource'] = loc_res_min['Machines']
        best_result['Exp. Avg. Wait Time'] = loc_res_min['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_res_min['BatchMaxAvg']

    return best_result
>>>>>>> main
