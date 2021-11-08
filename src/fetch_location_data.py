from numpy.lib.index_tricks import AxisConcatenator
from xlrd import Book
from numba import jit
from src.global_var import NUM_LOCATIONS
from src.settings import Settings
import numpy as np


COLUMN_NAMES = ['Likely or Exp. Voters', 'Eligible Voters', 'Ballot Length Measure']


def fetch_location_data(voting_config: Book) -> list:
    '''
        Fetches the locations sheet from the input xlsx as a dict.

        Params:
            voting_config (xlrd.Book) : input xlsx sheet.

        Returns:
            (list) : location sheet as a numpy array.
    '''
    location_sheet = voting_config.sheet_by_name('locations')
    data = np.empty((0, 3), int)
    arrival_mean_list = []
    

    for i in range(Settings.NUM_LOCATIONS + 1):
        if i == 0:
            continue

<<<<<<< HEAD
        data = np.append(data, np.array([location_sheet.row_values(i)[1:]]), axis=0) # [1:] drops ID column
        arrival_mean_list.append(data[i - 1][0] / Settings.POLL_OPEN / 60)
            
    arrival_mean = np.array(arrival_mean_list)
    location_data = np.concatenate((data, arrival_mean[:,None]),axis=1)
    
=======
        data = map(int, location_sheet.row_values(i)[1:])  # [1:] drops ID column
        location_data[i] = dict(zip(COLUMN_NAMES, data))

>>>>>>> main
    return location_data
