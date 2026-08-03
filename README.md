# osimpy

## Disclaimer

I developed this library for my own workflow, so I do not claim that it is the best way to work with OpenSim data in Python. It is simply a tool that I found useful for my own research, and I am sharing it in the hopes that it may be useful to others as well. I welcome feedback and contributions, but please keep in mind that this is a personal project and may not be suitable for all use cases.

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

#### CLI

#### Moco example

```python
from pathlib import Path

from osimpy import CoordinateReserveOptimalForceOverride, MocoInverseSettings


settings = MocoInverseSettings(
    name="walk11_inverse",
    model_path=Path("scaled_moco.osim"),
    coordinates_path=Path("BAA01_Baseline_Walk11_ik.mot"),
    external_loads_path=Path("BAA01_Baseline_Walk11_fp_setup.xml"),
    results_directory=Path("MocoInverse"),
    initial_time=9.025,
    final_time=9.435,
    replace_muscles_with_dgf=True,
    ignore_tendon_compliance=False,
    rigid_tendon_muscle_names=["L_GS", "R_GS"],
    dgf_fiber_damping=0.01,
    coordinate_reserve_optimal_force_overrides=[
        CoordinateReserveOptimalForceOverride(coordinate="sacrum_y", optimal_force=3.0),
        CoordinateReserveOptimalForceOverride(coordinate="hip_r_flx", optimal_force=0.5),
    ],
)

result = settings.run()
print(result.success, result.solution_file)
```

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
