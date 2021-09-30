# Project Globals

# NOTE: be careful with this init function, it should
# only be called ONCE when running anything.
# For unit tests, I have provided a way to overwrite these values
# with the overwrites dict.

def init(overwrites: dict = {}) -> None:
    '''
        Inits project globals to default values.
        Overwrite defaults using overwrites.

        Params:
            overwrites (dict) : global variables to overwrite.

        Returns:
            None.
    '''
    # deluxe version variable placeholders
    # 5:30, 6:30, 7:30
    global POLL_START
    POLL_START = 6.5
    global POLL_END
    POLL_END = 19.5

    # Add voter average arrivals in minutes
    global POLL_OPEN
    POLL_OPEN = POLL_END - POLL_START

    for var, value in overwrites.items():
        globals()[var] = value
