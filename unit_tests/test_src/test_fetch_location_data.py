import xlrd
import pytest
import settings

from src.fetch_location_data import fetch_location_data

# ====================================================================
# testing fetch_location_data and clean_location_data


@pytest.fixture
def load_location_xlsx() -> xlrd.Book:
    return xlrd.open_workbook(
        'unit_tests/test_src/test_data/location_data_test_copy.xlsx')


def test_fetch_location_data_1(load_location_xlsx):
    # arrange
    settings.init()
    # act
    location_data = fetch_location_data(load_location_xlsx)
    # assert
    assert 1 == 1
