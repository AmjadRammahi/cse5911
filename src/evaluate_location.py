import math
import logging
from pprint import pprint

from src.settings import Settings
from src.izgbs import izgbs

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

    loc_feas = []
    for key, value in loc_res.items():
        if value['Feasible'] == 1:
            loc_feas.append(value)

    if len(loc_feas) > 0:
        machines_value = []
        for result in loc_feas:
            machines_value.append(result['Machines'])
        
        # find fewest feasible machines
        mach_min = min(machines_value)

        # keep the feasible setup with the fewest number of machines
        loc_feas_min = {}
        for key, value in loc_res.items():
            if value['Machines'] == mach_min:
                loc_feas_min = value
                break

        # populate overall results with info for this location
        
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
        wait_time_min = min(max_avg)

        loc_res_min = loc_res[min_index]

        # for key, result in loc_res.items():
        #     if result['BatchMaxAvg'] == wait_time_min:
        #         loc_res_min = result
        #         break

        best_result['Resource'] = loc_res_min['Machines']
        best_result['Exp. Avg. Wait Time'] = loc_res_min['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_res_min['BatchMaxAvg']

    return best_result
