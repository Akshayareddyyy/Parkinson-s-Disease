# Contributing Guidelines

Thank you for your interest in contributing to the Parkinson's telemonitoring and machine learning benchmarking repository. This guide covers local environment setup, testing protocols, code conventions, and pull request procedures.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- Git
- Virtual environment tool (`venv` or `conda`)

### Local Setup
```bash
# Clone the repository
git clone https://github.com/Akshayareddyyy/Parkinson-s-Disease.git
cd Parkinson-s-Disease

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip and install frozen dependencies
python -m pip install --upgrade pip
pip install -r requirements-frozen.txt
```

---

## 2. Running Automated Tests

Before submitting any code changes, verify that the entire automated test suite passes cleanly:

```bash
# Run all unit tests
python -m unittest discover tests -v

# Run classical environment verification
python code/smoke_test.py

# Run quantum environment verification
python experiments/phase5/code/smoke_test.py
```

All contributions must maintain 100% test passing status. When introducing new models or preprocessing utilities, add corresponding unit tests under `tests/`.

---

## 3. Methodological & Scientific Constraints

This codebase implements a rigorous clinical telemonitoring benchmark. All contributors must strictly adhere to the following principles:

1. **Zero Data Leakage:**
   - Preprocessing scalers (e.g., `StandardScaler`, `MinMaxScaler`) must be fitted exclusively on the development/training partition (`D`).
   - Never compute mean, variance, or thresholds across the full dataset or across test partitions (`T`).
   - The binary classification threshold must be derived strictly as the median `motor_UPDRS` of the training partition.
2. **Subject-Disjoint Integrity:**
   - Patient records from the same `subject#` must never be split across training and evaluation partitions.
   - All evaluation must maintain the established 10-seed deterministic protocol.
3. **Preservation of Audited Outputs:**
   - Do not overwrite or modify frozen historical experiment logs, results CSV files, pre-trained checkpoints (`.pt`), or published figures.
   - New experiments or architectural variations should be created as separate, standalone modules.

---

## 4. Coding & Style Standards

- **Code Formatting:** Follow PEP 8 guidelines. Line length is configured to 120 characters in `pyproject.toml`.
- **Type Annotations:** Use Python type hints (`typing.List`, `typing.Dict`, `typing.Optional`, `typing.Tuple`) where practical for public interfaces.
- **Docstrings:** Document functions and classes using clear, descriptive docstrings explaining inputs, return types, and exceptions.
- **Imports:** Group imports by standard library, third-party libraries, and local project modules.

---

## 5. Development Workflow & Pull Requests

1. **Branch Naming:** Create feature branches using descriptive prefixes:
   - `feature/add-model-variant`
   - `fix/scaler-edge-case`
   - `docs/clarify-acoustic-formula`
   - `test/expand-coverage`
2. **Commit Messages:** Use clear, conventional commit messages:
   - `feat: add proximal loss term for federated training`
   - `fix: resolve division by zero in correlation calculation`
   - `docs: update dataset dictionary with sampling details`
   - `test: add unit tests for quantum expectation values`
3. **Pull Request Checklist:**
   - [ ] Automated unit test suite passes cleanly (`python -m unittest discover tests`).
   - [ ] No regression or leakage introduced to data partitions.
   - [ ] Relevant documentation updated for any interface changes.
