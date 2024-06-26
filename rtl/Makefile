VCS_HOME = /usr/local/packages/synopsys_2020/vcs/R-2020.12 # You'll probably have to change this
VCS_FLAGS =  -sverilog -partcomp=modautopart -notice -debug_access+all -timescale=1ns/1ns +lint=all

MAKO_SRC_DIR = mako_srcs
SEARCH_MAKO = $(addsuffix /*.sv.mako, $(MAKO_SRC_DIR))
MAKO_SRCS = $(wildcard $(SEARCH_MAKO))

TESTBENCHES = $(subst testbench/,,$(subst .sv,,$(wildcard testbench/*.sv)))

SV_SRC_DIR = sv_srcs
SV_SRCS = $(subst .mako,,$(subst $(MAKO_SRC_DIR),$(SV_SRC_DIR),$(MAKO_SRCS)))

MODEL ?= default.pickle.lzma
BUS_WIDTH ?= 64
COMPRESSED ?= 0
XPUT ?= -1
ifneq ($(COMPRESSED), 0)
	COMPRESS = --compressed
else
	COMPRESS = 
endif

.PHONY: say
say:
	echo $(MAKO_SRCS)
	echo $(TESTBENCHES)
	#echo $(SV_SRCS)
	for testbench in $(TESTBENCHES); do \
		echo $$(echo $$testbench | sed 's/\.sv//'); \
	done	

#all: template compile run

.PHONY: template
template: $(MAKO_SRCS)
	python3 build/make_rtl.py $(MAKO_SRCS) --model $(MODEL) --bus_width $(BUS_WIDTH) --throughput $(XPUT) $(COMPRESS)

.PHONY: compile
compile: template
	$(VCS_HOME)/bin/vcs -full64 ${SV_SRCS} $(VCS_FLAGS) -l compile.log 

$(TESTBENCHES): %: testbench/%.sv compile
	mkdir -p testbench_objs
	$(VCS_HOME)/bin/vcs -full64 ${SV_SRCS} $< -o testbench_objs/$@ $(VCS_FLAGS) -l compile_$@.log

.PHONY: testbenches
testbenches: $(TESTBENCHES)

.PHONY: clean
clean:
	rm sv_srcs/*.sv sv_srcs/*.svh
	rm -r testbench_objs/*
	rm -r partitionlib/

.PHONY: sterilize
sterilize: clean
	rm -r csrc/
	rm -r diag_dir/
	rm -r testbench_objs/
	rm -r simv.daidir/
	rm *.vpd *.key

