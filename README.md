# ULEEN (Ultra Low Energy Edge Networks)

Code to accompany the paper:

**ULEEN: A Novel Architecture for Ultra Low-Energy Edge Neural Networks**, Zachary Susskind, Aman Arora, Igor D. S. Miranda, Alan T. L. Bacellar, Luis A. Q. Villon, Rafael F. Katopodis, Leandro S. de Araújo, Diego L. C. Dutra, Priscila M. V. Lima, Felipe M. G. França, Mauricio Breternitz Jr., and Lizy K. John

*Published in [ACM Transactions on Architecture and Code Optimization (TACO)](https://dl.acm.org/doi/10.1145/3629522)*

For questions, please contact ZSusskind *(at)* gmail *(dot)* com (or utexas *(dot)* edu).

**I've graduated now and am not actively maintaining this repository; however, you should check out the [DWN](https://github.com/alanbacellar/DWN) repository, which *is* actively maintained and is superior to ULEEN in every way.**

### Beta Release
This code has been rewritten and cleaned up to remove the considerable amount of "research-grade" jank that was present in my development repository. However, there may be some oversights that were made in this process. Please bear with me on this, and contact me if you encounter any bugs or strange behaviors.

**Known issues:**
 - The code for preprocessing the ToyADMOS and KWS datasets is currently absent. This is partially due to licensing concerns, since I pulled some of the code for this from other repositories. In the meantime, you can download preprocessed datasets directly from [my personal website](https://zsknd.com/misc/preprocessed_data/).
 - JSON files to replicate specific models from the paper are missing. In the mean time, you can take a look at the provided `example.json` file.
 - I seem to have introduced a regression in the unused input pruning in `finalize_model.py` when I was cleaning up the code. I've disabled this for now. 

## Installation
This code was written for Python 3.8.10; other versions are untested. For compatibility, I suggest using [PyEnv](https://github.com/pyenv/pyenv) if needed.

To install the necessary packages:
 1. Set up a virtual environment: `python3 -m venv env`
 2. Source the virtual environment: `source env/bin/activate`
 3. Install the packages: `pip3 install -r requirements.txt`

## Training
Training code is present in the `software_model` directory. There are three steps:
1. Training the initial model (`train_model.py`)
2. Pruning the trained model (`prune_model.py`)
3. Preparing the pruned model for inference (`finalize_model.py`)

This can all be done in a single command using the `full_run.py` script, which takes a JSON configuration file as input. Take a look at `example.json` for syntax.

## Inference
 The RTL templates, testbench, and build scripts are present in the `rtl` directory.
 
 To build RTL for a model, run `make template`. There are a few flags you can provide:
 - `VCS_HOME`: Path to your Synopsys VCS installation (only needed if you're using the VCS testbench)
 - `MODEL`: Path to finalized (`.pickle.lzma`) model file
 - `BUS_WIDTH`: Input bus width for model
 - `COMPRESSED`: Whether inputs to the accelerator are compressed and should therefore be decompressed on-chip
 - `XPUT`: Target throughput of the design. Leaving this at the default (-1) will instantiate just enough hash and decompression functional units to avoid being compute bound.

RTL templates are in the `mako_srcs/` directory, and generated RTL is placed in the `sv_srcs/` directory. The testbench is available at `testbench/test_functional_correctness.sv`, and can be built using `make testbenches` using the flags described above.
