import settings
from xlrd import Book

from src.data_set import DataSet


def fetch_location_data(voting_config: Book) -> DataSet:
    '''
        Fetches the locations sheet from the input xlsx as a DataSet.

        Params:
            voting_config (xlrd.Book) : input xlsx sheet.

        Returns:
            (DataSet) : location sheet as a DataSet.
    '''
    location_sheet = voting_config.sheet_by_name(u'locations')

    location_data = []

    for i in range(location_sheet.nrows):
        if i == 0:
            columns = {
                column_name: j
                for j, column_name in enumerate(location_sheet.row_values(
                    i,
                    start_colx=0,
                    end_colx=None
                ))
            }
        else:
            location_data.append(location_sheet.row_values(i))

    location_data = DataSet(location_data, columns)

    clean_location_data(location_data)

    return location_data


def clean_location_data(location_data: DataSet) -> None:
    '''
        Updates the location_data read in from the input xlsx.

        Params:
            location_data (DataSet) : locations tab from input xlsx.

        Returns:
            None.
    '''
    # sort voting location for optimization
    location_data.sort_values(
        ['Likely or Exp. Voters', 'Eligible Voters', 'Ballot Length Measure'],
        ascending=False
    )

    # create location ID specific to new sort order
    location_data['Loc_ID'] = [int(row[0] + 1) for row in location_data]
    # location_data['Loc_ID'] = (location_data.index + 1).astype('int')

    # convert columns to numeric so they can be used for calculations
    location_data['Likely or Exp. Voters'] = list(map(int, location_data['Likely or Exp. Voters']))
    location_data['Eligible Voters'] = list(map(int, location_data['Eligible Voters']))
    location_data['Ballot Length Measure'] = list(map(int, location_data['Ballot Length Measure']))

    # convert ID to int
    location_data['ID'] = list(map(int, location_data['ID']))

    location_data['Arrival_mean'] = [
        v / settings.POLL_OPEN / 60
        for v in location_data['Likely or Exp. Voters']
    ]


# def old_fetch_location_data(voting_config: Book) -> list:
#     location_sheet = voting_config.sheet_by_name(u'locations')

#     location_data = []

#     for i in range(location_sheet.nrows):
#         location_data.append(
#             location_sheet.row_values(
#                 i,
#                 start_colx=0,
#                 end_colx=None
#             )
#         )

#     return location_data


# def old_clean_location_data(location_data: list) -> None:
#     import pandas as pd

#     voting_machines = pd.DataFrame(location_data)
#     voting_machines.columns = voting_machines.iloc[0]
#     voting_machines = voting_machines.iloc[1:]

#     # sort voting location for optimization
#     voting_machines.sort_values(
#         ['Likely or Exp. Voters', 'Eligible Voters', 'Ballot Length Measure'],
#         ascending=False,
#         inplace=True
#     )

#     # create location ID specific to new sort order
#     voting_machines['Loc_ID'] = (voting_machines.index + 1).astype('int')

#     # convert columns to numeric so they can be used for calculations
#     voting_machine_nums = [
#         'Likely or Exp. Voters',
#         'Eligible Voters',
#         'Ballot Length Measure'
#     ]
#     voting_machines[voting_machine_nums] = voting_machines[voting_machine_nums].astype('int')

#     # convert ID to int
#     voting_machines['ID'] = voting_machines['ID'].astype('int')

#     voting_machines['Arrival_mean'] = \
#         voting_machines['Likely or Exp. Voters'] / 13 / 60

#     return voting_machines
