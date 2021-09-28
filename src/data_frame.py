from typing import Union


class DataFrame:

    def __init__(self, data: list = [], columns: dict = {}):
        self.data = data
        self.columns = columns

    def append_row(self, row: list) -> None:
        '''
            Appends a row of data to this.data.

            Params:
                row (list) : row of data.

            Returns:
                None.
        '''
        self.data.append(row)

    def sort_values(self, by: Union[str, list], ascending: bool = True) -> None:
        for key in by[::-1]:
            self.data = sorted(
                self.data,
                key=lambda x: x[self.columns[key]],
                reverse=not ascending
            )

    def __getitem__(self, i: Union[str, int]) -> list:
        pass

    def __iter__(self):
        for row in self.data:
            yield row

    def __len__(self):
        return len(self.data)
