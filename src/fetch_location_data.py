from xlrd import Book

from src.settings import Settings


COLUMN_NAMES = ['Likely or Exp. Voters', 'Eligible Voters', 'Ballot Length Measure']


def fetch_location_data(voting_config: Book) -> dict:
    '''
        Fetches the locations sheet from the input xlsx as a dict.

        Params:
            voting_config (xlrd.Book) : input xlsx sheet.

        Returns:
            (dict) : location sheet as a dict.
    '''
    location_sheet = voting_config.sheet_by_name(u'locations')
    location_data = {}
    

    for i in range(Settings.NUM_LOCATIONS + 1):
        if i == 0:
            continue

        data = map(int, location_sheet.row_values(i)[1:])  # [1:] drops ID column
        location_data[i] = dict(zip(COLUMN_NAMES, data))
        location_data[i].update({
            'Arrival Mean': location_data[i]['Likely or Exp. Voters'] / Settings.POLL_OPEN / 60
        })
    
    return location_data
