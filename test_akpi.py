from src.settings import default_settings
from src.izgbs import izgbs


if __name__ == '__main__':

    settings = default_settings()
    settings['SERVICE_REQ'] = 30

    # should print back resource apportionments of 2, 27, 53
    # getting 2, 29, 26, this matches the old implementation that did not include AKPI
    print(izgbs({'Likely or Exp. Voters': 100, 'Eligible Voters': 100, 'Ballot Length Measure': 1}, settings))
    print(izgbs({'Likely or Exp. Voters': 2500, 'Eligible Voters': 2500, 'Ballot Length Measure': 1}, settings))
    print(izgbs({'Likely or Exp. Voters': 5000, 'Eligible Voters': 5000, 'Ballot Length Measure': 1}, settings))
