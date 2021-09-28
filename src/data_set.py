from typing import Union


class DataSet:

    def __init__(self, data: list = [], columns: dict = {}):
        self.data = data
        self.columns = columns
        self.shape = [len(data), len(data[0])]

        self._validate_shape()

    def _validate_shape(self) -> None:
        '''
            Double checks all entries in self.data match self.shape.

            Returns:
                None.

            Exceptions:
                (ValueError) : if inconsistent row lengths in self.data.
        '''
        for row in self.data:
            if len(row) != self.shape[1]:
                raise ValueError(
                    f'DataSet data error: Input row lengths not consistent')

    def _validate_list(self, li: list, size: int) -> None:
        '''
            Double checks the list being assigned is a list of correct shape.

            Params:
                li (list) : list to validate,
                size (int) : expected size for the list.

            Returns:
                None.

            Exceptions:
                (ValueError) : of the list is not a list or not of correct size.
        '''
        if not (isinstance(li, list) and len(li) == size):
            raise ValueError(
                f'DataSet assignment error: '
                f'Input {li} is not a list of length {size}')

    def _validate_key(self, key: Union[str, int, list]) -> None:
        '''
            Checks if the given key is valid for this DataSet.

            Params:
                key (Union[str, int, list]) : DataSet key,
                should_return (bool) : whether to return True/False or raise.

            Returns:
                None.

            Exceptions:
                (ValueError) : if the key is not a member of this DataSet.
        '''
        if isinstance(key, str):
            if key not in self.columns:
                raise ValueError('TODO')
        elif isinstance(key, int):
            if key >= self.shape[0]:
                raise ValueError('TODO')
        elif isinstance(key, list):
            if all(isinstance(k, int) for k in key):
                for k in key:
                    if k >= self.shape[0]:
                        raise ValueError('TODO')
            elif all(isinstance(k, str) for k in key):
                for k in key:
                    if k not in self.columns:
                        raise ValueError('TODO')
            else:
                raise ValueError(
                    f'DataSet key error: '
                    f'Key must be all ints or all strings, recieved {key}')
        else:
            raise ValueError(
                f'DataSet key error: '
                f'Recieved key of unknown type \'{type(key)}\'')

    def append_row(self, row: list) -> None:
        '''
            Appends a row of data to this.data.

            Params:
                row (list) : row of data.

            Returns:
                None.
        '''
        self._validate_list(row, self.shape[1])

        self.data.append(row)
        self.shape[0] += 1

    def sort_values(self, by: Union[str, list], ascending: bool = True) -> None:
        '''
            Sorts the DataSet by one or more columns.

            Params:
                by (Union[str, list]) : columns to sort the DataSet by,
                ascending (bool) : True for ascending, False for descending.

            Returns:
                None.
        '''
        if isinstance(by, str):
            by = [by]

        for key in by[::-1]:
            self.data = sorted(
                self.data,
                key=lambda x: x[self.columns[key]],
                reverse=not ascending
            )

    def head(self) -> None:
        '''
            Prints the DataSet to stdout.

            Returns:
                None.
        '''
        print([c for c in self.columns])
        print('[')

        for row in self.data:
            print('  ', row)

        print(']')

    def __getitem__(self, key: Union[str, int, list]) -> list:
        self._validate_key(key)

        if isinstance(key, str):
            # returning a column
            return [row[self.columns[key]] for row in self.data]
        elif isinstance(key, int):
            # returning a row
            return self.data[key]
        elif isinstance(key, list):
            if isinstance(key[0], str):
                # returning multiple columns
                return [
                    [
                        row[self.columns[k]]
                        for row in self.data
                    ]
                    for k in key
                ]
            elif isinstance(key[0], int):
                # returning multiple rows
                return [self.data[k] for k in key]

        raise ValueError(
            f'DataSet key error: '
            f'Recieved key of unknown type \'{type(key)}\'')

    def __setitem__(self, key: Union[str, int, list], value: list) -> None:
        if isinstance(key, str):
            self._validate_list(value, self.shape[0])

            if key in self.columns:
                # assigning to an existing column
                for i, row in enumerate(self.data):
                    row[self.columns[key]] = value[i]
            else:
                # assigning a new column
                self.columns[key] = len(self.columns)
                self.shape[1] += 1

                for i, row in enumerate(self.data):
                    row.append(value[i])

        elif isinstance(key, int):
            self._validate_list(value, self.shape[1])

            # assigning a new row
            self.data[key] = value

        elif isinstance(key, list):
            if all(isinstance(k, str) for k in key):
                # assigning 1 or more new columns
                for k in key:
                    self._validate_list(value[k], self.shape[1])

                    raise NotImplementedError()
            elif all(isinstance(k, int) for k in key):
                # assigning 1 or more new rows
                for k in key:
                    self._validate_list(value[k], self.shape[1])

                    self.data[k] = value[k]
            else:
                raise ValueError(
                    f'DataSet key error: '
                    f'Key must be all ints or all strings, recieved {key}')
        else:
            raise ValueError(
                f'DataSet key error: '
                f'Recieved key of unknown type \'{type(key)}\'')

    def __iter__(self):
        for row in self.data:
            yield row

    def __len__(self):
        return self.shape[0]
