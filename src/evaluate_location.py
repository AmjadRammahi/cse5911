import math

from src.izgbs import izgbs


def evaluate_location(inputs: tuple) -> dict:
    '''
        Runs IZGBS on a specified location.

        Params:
            inputs (tuple) : location data, settings.

        Returns:
            (dict) : results.
    '''
    location_data, settings = inputs

    best_result = {}

    start_val = math.ceil((settings['MAX_MACHINES'] - 1) / 2)

    loc_res = izgbs(
        settings['MAX_MACHINES'],
        start_val,
        settings['MIN_MACHINES'],
        location_data,
        settings
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

        loc_res_min = loc_res[min_index]

        best_result['Resource'] = loc_res_min['Machines']
        best_result['Exp. Avg. Wait Time'] = loc_res_min['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_res_min['BatchMaxAvg']

    return best_result
