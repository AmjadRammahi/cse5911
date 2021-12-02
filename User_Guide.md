# **User Guide**
This software can be run in two different ways:

1. Through the Excel file (after performing simple initial set-up)
2. On the command line by running the Python scripts directly

# Excel Usage
First run the included makefile to create the `.exe` windows executables. Open a terminal and type the following command to run the makefile:

```
make
```

Next, use Excel to open voting_exce.xlsm and press either the allocation or apportionment button to run the associated program. The simulation output will be automatically populated in the Excel file.

# Command Line Usage
Open a terminal and run the following command to install all of the necessary dependencies for the software.

```
pip install -r requirements.txt
```

Next, the software can be executed with either one of the following commands:

```
python3 apportionment.py voting_excel.xlsx
```


```
python3 allocation.py voting_excel.xlsx
```

# Changing Program Settings
All settings for both allocation and apportionment are controlled from the Settings page within the voting_excel.xlsm file. Any settings changed and saved on this page will be effective on the next run of the software.
