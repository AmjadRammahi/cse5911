# **CSE5911 - ENOVA**

## Contributers
* Amjad Rammahi
* Zhiren Xu
* Collin Wright
* Luke Howard
* Tian Liang - Original Contributer
* Jennifer Heider - Original Contributer

## Project Overview

[Efficient Near-Optimal Voting Allocation.PDF](Efficient&#32;Near-Optimal&#32;Voting&#32;Allocation.pdf)

# Resources

### *[Determining resource requirements for elections using indifference-zone generalized binary search](https://www.sciencedirect.com/science/article/pii/S0360835219307120)*

ScienceDirect paper, covers IZGBS.

# Setup
```
pip install -r requirements.txt
```

# Usage
```
python3 apportionment.py voting_excel.xlsx
```

# Alternate Usage
```
make
```
Open voting_excel.xlsm - press either the allocation or apportionment buttons


# Notes
* Initial runtime of the code was 1622.43 seconds (benchmarked on OSC).

* Apportionment - Infinite resources per locations - goal: min resources that meet specified wait time req.
* Allocation - Fixed number of resources, fixed number of locations - goal: best distribute resources to minimize total wait times.