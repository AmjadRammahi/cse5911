import math
import logging
from numba import njit
from src.settings import Settings
from src.izgbs import izgbs


def evaluate_location(params: list) -> 'best_result_dict':
    '''
        Runs IZGBS on a specified location, return results in dict best_results.

        Params:
            params (list) : location data dict, location id.

        Returns:
            best_result: a dict contain all field that going to fill in result_df.
    '''
    best_result = {}
    location_data, location_id = params

    logging.info(f'Starting Location: {location_id}')

    # Placeholder, use a different start value for later machines?
    # NOTE/TODO: apportionment vs appointment - Collin
    if Settings.MIN_ALLOC_FLG:
        start_val = math.ceil(
            (Settings.MAX_MACHINES - Settings.MIN_ALLOC) / 2) + Settings.MIN_ALLOC
        sas_alpha_value = Settings.ALPHA_VALUE / math.log2(Settings.MAX_MACHINES - Settings.MIN_ALLOC)

        loc_res = izgbs(
            location_id,
            Settings.MAX_MACHINES,
            start_val,
            Settings.MIN_ALLOC,
            sas_alpha_value,
            location_data
        )
    else:
        start_val = math.ceil((Settings.MAX_MACHINES - 1) / 2)
        sas_alpha_value = Settings.ALPHA_VALUE / math.log2(Settings.MAX_MACHINES - Settings.MIN_ALLOC)

        loc_res = izgbs(
            location_id,
            Settings.NUM_MACHINES,
            start_val,
            2,
            sas_alpha_value,
            location_data
        )

    loc_feas = loc_res[loc_res['Feasible'] == 1].copy()

    if not loc_feas.empty:
        # calculate fewest feasible machines
        mach_min = loc_feas['Machines'].min()

        # keep the feasible setup with the fewest number of machines
        loc_feas_min = loc_feas[loc_feas['Machines'] == mach_min].copy()

        # populate overall results with info for this location
        best_result['i'] = location_id
        best_result['Resource'] = mach_min
        best_result['Exp. Avg. Wait Time'] = loc_feas_min.iloc[0]['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_feas_min.iloc[0]['BatchAvgMax']

        return best_result
