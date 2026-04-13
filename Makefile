# 確率換算表（/ptable）: ../andersan-train の *.table.feather を tables/ に同期する。
#   手動: make ptables
PTABLE_STEMS := andersan0_1 andersan0_1_1 andersan0_2 andersan0_2_1 andersan1

.PHONY: all ptables

all: standards.json andersan0_1.py.best.keras

standards.json: ../andersan-train/datatype3/standards.json
	cp $< $@

andersan0_1.py.best.keras: ../andersan-train/andersan0_1.py.best.keras
	cp $< $@

ptables: $(addprefix tables/,$(addsuffix .table.feather,$(PTABLE_STEMS)))

tables/%.table.feather: ../andersan-train/%.table.feather
	@mkdir -p $(dir $@)
	cp $< $@
