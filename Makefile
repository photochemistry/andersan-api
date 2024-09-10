all: standards.json andersan0_1.py.best.keras
standards.json: ../andersan-train/datatype3/standards.json
	cp $< $@

andersan0_1.py.best.keras: ../andersan-train/andersan0_1.py.best.keras
	cp $< $@
