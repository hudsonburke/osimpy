# osimpy

## Description

This repository contains tools for working with OpenSim models and simulations in Python. It includes functionality for manipulating OpenSim models, setting up and running simulations, and analyzing results. It strives to be Pythonic and user-friendly, making it easier for researchers and developers to work with OpenSim data by providing type hints and descriptions for IDE support.

## Quickstart

### Installation

If not already installed, install:

- [git](https://git-scm.com/install/)
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  - Miniconda is sufficient, but any anaconda installation will work

``` shell
git clone https://github.com/hudsonburke/osimpy.git
cd osimpy

conda env create -f environment.yml
conda activate osimpy

python -m pip install -e .
```

### Usage

## Contributing

### TODO

- [ ] Define metadata schemas for files
- [ ] Clean up osim_graph
- [ ] Implement other moco functionality
- [ ] Create tests
- [ ] OpenSim Python bindings using nanobind or pybind11
  - Doing this would essentially supplant most of this repo's functionality
- [ ] Switch to uv for dependency management
  - Currently waiting for opensim bindings to be easily available
  - Pyopensim doesn't quite work

## Citing

If you use osimpy in your research, please cite:

```bibtex

```
