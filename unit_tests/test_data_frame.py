import pytest
from src.data_frame import DataFrame

import pandas as pd  # TODO: just for debugging


def test_data_frame_init_1():
    # arrange
    df = DataFrame(
        [
            ['John', 2],
            ['Jim', 9],
            ['Jason', 1]
        ],
        columns={
            'Names': 0,
            'Values': 1
        }
    )
    # assert
    assert df.data == [['John', 2], ['Jim', 9], ['Jason', 1]]
    assert df.columns == {'Names': 0, 'Values': 1}


def test_data_frame_sort_values_descending_1():
    # arrange
    df = DataFrame(
        [
            ['John', 2],
            ['Jim', 9],
            ['Jason', 1]
        ],
        columns={
            'Names': 0,
            'Values': 1
        }
    )
    # act
    df.sort_values(
        by=['Values'],
        ascending=False
    )
    # assert
    assert df.data == [
        ['Jim', 9],
        ['John', 2],
        ['Jason', 1]
    ]


def test_data_frame_sort_values_descending_2():
    # arrange
    df = DataFrame(
        [
            ['John', 2, 1, 3],
            ['Jim', 9, 2, 2],
            ['Jason', 1, 3, 1]
        ],
        columns={
            'Names': 0,
            'Values1': 1,
            'Values2': 2,
            'Values3': 3
        }
    )
    # act
    df.sort_values(
        by=['Values1', 'Values2', 'Values3'],
        ascending=False
    )
    # assert
    assert df.data == [
        ['Jim', 9, 2, 2],
        ['John', 2, 1, 3],
        ['Jason', 1, 3, 1]
    ]


def test_data_frame_sort_values_descending_3():
    # arrange
    df = DataFrame(
        [
            ['John', 2, 1, 3],
            ['Jim', 9, 2, 2],
            ['Jason', 1, 3, 1]
        ],
        columns={
            'Names': 0,
            'Values1': 1,
            'Values2': 2,
            'Values3': 3
        }
    )
    # act
    df.sort_values(
        by=['Names', 'Values1', 'Values2', 'Values3'],
        ascending=False
    )
    # assert
    assert df.data == [
        ['John', 2, 1, 3],
        ['Jim', 9, 2, 2],
        ['Jason', 1, 3, 1]
    ]


def test_data_frame_sort_values_ascending_1():
    # arrange
    df = DataFrame(
        [
            ['John', 2],
            ['Jim', 9],
            ['Jason', 1]
        ],
        columns={
            'Names': 0,
            'Values': 1
        }
    )
    # act
    df.sort_values(
        by=['Values'],
        ascending=True
    )
    # assert
    assert df.data == [
        ['Jason', 1],
        ['John', 2],
        ['Jim', 9]
    ]


def test_data_frame_sort_values_ascending_2():
    # arrange
    df = DataFrame(
        [
            ['John', 2, 1, 3],
            ['Jim', 9, 2, 2],
            ['Jason', 1, 3, 1]
        ],
        columns={
            'Names': 0,
            'Values1': 1,
            'Values2': 2,
            'Values3': 3
        }
    )
    # act
    df.sort_values(
        by=['Values1', 'Values2', 'Values3'],
        ascending=True
    )
    # assert
    assert df.data == [
        ['Jason', 1, 3, 1],
        ['John', 2, 1, 3],
        ['Jim', 9, 2, 2]
    ]


def test_data_frame_sort_values_ascending_3():
    # arrange
    df = DataFrame(
        [
            ['John', 2, 1, 3],
            ['Jim', 9, 2, 2],
            ['Jason', 1, 3, 1]
        ],
        columns={
            'Names': 0,
            'Values1': 1,
            'Values2': 2,
            'Values3': 3
        }
    )
    # act
    df.sort_values(
        by=['Names', 'Values1', 'Values2', 'Values3'],
        ascending=True
    )
    # assert
    assert df.data == [
        ['Jason', 1, 3, 1],
        ['Jim', 9, 2, 2],
        ['John', 2, 1, 3]
    ]

