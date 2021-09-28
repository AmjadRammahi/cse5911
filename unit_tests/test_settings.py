import sys
import settings  # NOTE: don't remove this, the sys.modules.pop() will fail

# ====================================================================
# Testing settings which houses project globals

# NOTE: theres some subtle nuance here. When we call settings.init, settings adds
# a whole bunch of new variables to globals(). These globals will persist between tests.
# Flushing the settings globals() using sys.modules.pop().
# NOTE: importlib.reload(settings) does not flush the globals.


def test_settings_uninitialized_1():
    # arrange
    sys.modules.pop('settings')
    import settings
    # assert
    assert hasattr(settings, 'POLL_START') is False


def test_settings_default_1():
    # arrange
    sys.modules.pop('settings')
    import settings
    # act
    settings.init()
    # assert
    assert settings.POLL_START == 6.5


# NOTE: testing that the POLL_START globally was flushed from the prior test
def test_settings_uninitialized_2():
    # arrange
    sys.modules.pop('settings')
    import settings
    # assert
    assert hasattr(settings, 'POLL_START') is False


def test_settings_overwrites_1():
    # arrange
    sys.modules.pop('settings')
    import settings
    # act
    settings.init({
        'POLL_START': 7.5
    })
    # assert
    settings.POLL_START == 7.5
