from src.settings import Settings, reset_settings
from allocation import allocation


def test_allocation_runs_without_error():
    Settings.NUM_LOCATIONS = 2
    Settings.BATCH_SIZE = 1
    Settings.NUM_REPLICATIONS = 3
    Settings.NUM_BATCHES = 3 // 1
    result = allocation(
        {
            1: {
                "Likely or Exp. Voters": 50,
                "Eligible Voters": 100,
                "Ballot Length Measure": 2
            },
            2: {
                "Likely or Exp. Voters": 50,
                "Eligible Voters": 100,
                "Ballot Length Measure": 2
            }
        },
        10,
        2
    )
    reset_settings()

    assert type(result) is dict


def test_allocation_equal_for_equal_locations():
    Settings.NUM_LOCATIONS = 2
    Settings.BATCH_SIZE = 1
    Settings.NUM_REPLICATIONS = 3
    Settings.NUM_BATCHES = 3 // 1
    result = allocation(
        {
            1: {
                "Likely or Exp. Voters": 100,
                "Eligible Voters": 200,
                "Ballot Length Measure": 4
            },
            2: {
                "Likely or Exp. Voters": 100,
                "Eligible Voters": 200,
                "Ballot Length Measure": 4
            }
        },
        100,
        96
    )
    reset_settings()

    assert result[1]['Resource'] == result[2]['Resource']


def test_allocation_uses_expected_num_machines():
    Settings.NUM_LOCATIONS = 2
    Settings.BATCH_SIZE = 1
    Settings.NUM_REPLICATIONS = 3
    Settings.NUM_BATCHES = 3 // 1
    result = allocation(
        {
            1: {
                "Likely or Exp. Voters": 50,
                "Eligible Voters": 100,
                "Ballot Length Measure": 2
            },
            2: {
                "Likely or Exp. Voters": 50,
                "Eligible Voters": 100,
                "Ballot Length Measure": 2
            }
        },
        10,
        2
    )
    reset_settings()

    machines_used = sum(v['Resource'] for v in result.values())

    assert abs(machines_used - 10) <= 2
