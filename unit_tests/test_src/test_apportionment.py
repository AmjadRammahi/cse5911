import pytest
import src.global_var as global_var
from src.settings import Settings
from apportionment import apportionment


# Super simple test case to ensure apportionment.py runs without error (this
# test does not validate results).
def test_apportion_no_error():
    global_var.BATCH_SIZE = 1
    # Note: setting replications < 3 will cause the test to fail at line #136
    # in src/izgbs.py
    global_var.NUM_REPLICATIONS = 3
    global_var.NUM_LOCATIONS = 1
    Settings.NUM_LOCATIONS = 1
    # Set-up simple location_data object
    location_data = {
        1: {
            "Likely or Exp. Voters": 50,
            "Eligible Voters": 100,
            "Ballot Length Measure": 2
        }
    }
    # No assertion needed, test will fail if and only if apportionment throws
    # an error
    apportionment(location_data)
