# 💾 Floppy: FLOP Analysis and Counting Framework

Floppy is a framework for analyzing and counting floating-point operations (FLOPs) in Python code, with special focus on numerical computations using NumPy.

## Features

- **Runtime FLOP Counting**: Track FLOPs in your code as it executes
  - Transparent array operation tracking
  - Support for high-level NumPy operations
  - Thread-safe operation
  - Selective monitoring of code paths

- **Static Analysis**: Analyze potential FLOP operations in your codebase
  - Identify potential FLOP-contributing functions
  - Get insights before execution

## Installation

For `monty_lab` conda environment, please follow instructions from [Thousand Brains Project documentation](https://thousandbrainsproject.readme.io/docs/getting-started) and run:

```bash
conda create --clone tbp.monty -n monty_lab
conda activate monty_lab
cd path/to/monty_lab/floppy
pip install -e .
```

## Quick Start

```python
from floppy.counting.base import FlopCounter

with FlopCounter() as counter:
    result = np.matmul(a, b)
    print(f"FLOPs: {counter.flops}")
```

For analyzing an entire codebase:

```bash
python run_static_analysis.py --dir path/to/analyze
```

## Documentation

- [Getting Started](docs/user_guide/getting_started.md)
- [User Guide](docs/user_guide/index.md)
- [API Reference](docs/api/index.md)

## Contributing

Contributions are welcome! Please follow the [Thousand Brains Project Contributing Guide](https://thousandbrainsproject.readme.io/docs/contributing).

## License

The MIT License. See the [LICENSE](LICENSE) for details.
