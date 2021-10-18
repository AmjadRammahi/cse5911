import math
import logging

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

    loc_feas = loc_res[loc_res['Feasible'] == 1].copy()

    if not loc_feas.empty:
        # find fewest feasible machines
        mach_min = loc_feas['Machines'].min()

        # keep the feasible setup with the fewest number of machines
        loc_feas_min = loc_feas[loc_feas['Machines'] == mach_min].copy()

        # populate overall results with info for this location
        best_result['Resource'] = mach_min
        best_result['Exp. Avg. Wait Time'] = loc_feas_min.iloc[0]['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_feas_min.iloc[0]['BatchMaxAvg']
    else:
        # no feasible setups, find lowest wait time (should work out to be max machines allowed)
        wait_time_min = loc_res['BatchMaxAvg'].min()

        loc_res_min = loc_res[loc_res['BatchMaxAvg'] == wait_time_min]

        best_result['Resource'] = loc_res_min.iloc[0]['Machines']
        best_result['Exp. Avg. Wait Time'] = loc_res_min.iloc[0]['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_res_min.iloc[0]['BatchMaxAvg']

    return best_result
