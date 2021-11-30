.PHONY: all apportionment allocation

all:
	apportionment
	allocation

apportionment:
	pyinstaller --noconfirm --onefile --console  "apportionment.py" --distpath ./
	del apportionment.spec

allocation:
	pyinstaller --noconfirm --onefile --console  "allocation.py" --distpath ./
	del allocation.spec
