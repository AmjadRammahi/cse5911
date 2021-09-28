from src.data_set import DataSet

# ====================================================================
# DataSet.__init__


def test_data_set_init_1():
    # arrange
    ds = DataSet(
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
    assert ds.data == [['John', 2], ['Jim', 9], ['Jason', 1]]
    assert ds.columns == {'Names': 0, 'Values': 1}
