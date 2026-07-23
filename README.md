# osimpy

## Description

Pythonic wrapper and tools for working with OpenSim models and analyses. Provides functionality for manipulating OpenSim models, setting up and running simulations, and analyzing results, with type hints and descriptions for IDE support.

## Quickstart

``` shell
git clone https://github.com/hudsonburke/osimpy.git
cd osimpy

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Modules

| Module | Description |
|--------|-------------|
| `io` | Read/write OpenSim file formats (.sto, .mot, .trc, external loads XML) |
| `tools` | Pydantic-based wrappers for OpenSim tools (Scale, IK, ID, CMC) |
| `osim_graph` | Graph-based muscle path analysis across joint configurations |
| `moco` | MocoInverse solver wrapper |
| `utils` | Unit conversion, actuator/task file helpers |

## Development

```shell
uv sync --group dev
```

## Notes

- OpenSim 4.6 wheels are available for Python 3.11–3.13.
- The `OsimGraph` module uses multiprocessing for muscle path analysis; set `OMP_NUM_THREADS=1` for deterministic behavior.
