# MOFGen: Cross-Platform Fix, Test Suite, CI, and MP Data Extraction

**Date:** 2026-04-06
**Status:** Approved

## Problem

Users report `pixi install` failures on macOS and Ubuntu due to invalid `pixi.toml` syntax (unexpected keys, ambiguous version specifiers). The latest commit (a422da0) partially addressed this but the repo still has:

- No CPU-only installation path (CUDA forced on Linux/Windows)
- TensorFlow cross-platform resolution issues
- Conflicting/broken `[tool.pixi.*]` sections in `pyproject.toml`
- Zero tests
- No CI to catch regressions
- No Materials Project data extraction script

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| GPU support | Pixi features: default CPU + opt-in `cuda` | Most portable, standard pixi pattern |
| TensorFlow | Keep as-is, make it work everywhere | Used in cif_to_tfrecord.py, user preference |
| Config source of truth | `pixi.toml` only | `pyproject.toml` pixi sections are broken |
| Test depth | Unit + mocked integration + CI smoke | Full coverage without external deps |
| MP data script | Env-var API key, full download + query | Secure, flexible |

## Section 1: pixi.toml Fix & Feature Split

### Default (CPU) Environment
- Platforms: linux-64, osx-64, osx-arm64, win-64
- Channels: conda-forge, pytorch
- Python 3.11
- PyTorch CPU, TensorFlow, all other deps
- No CUDA dependencies

### CUDA Feature
- Adds `cuda-version = "12.1.*"` on linux-64 and win-64
- Users opt in: `pixi install -e cuda`
- GPU-enabled PyTorch

### TensorFlow Cross-Platform Strategy
- Use `tensorflow` from conda-forge (supports linux-64, osx-64, osx-arm64, win-64 as of recent builds)
- If conda-forge resolution fails on a platform, fall back to `tensorflow` as a pypi-dependency for that platform target
- Pin to a version range known to work across platforms (>=2.15,<3)

### pyproject.toml Cleanup
- Remove all `[tool.pixi.*]` sections
- Keep standard `[project]` metadata only

### requirements.txt
- Keep as supplemental pip reference (not used by pixi)

## Section 2: Test Suite

```
tests/
├── conftest.py                    # Shared fixtures (mock CIF data, temp dirs)
├── test_diffusion/
│   ├── test_cif_to_tfrecord.py    # TFRecord conversion (file I/O, parsing)
│   ├── test_dataset.py            # MOFDataset loading, shape validation
│   ├── test_model.py              # DiT architecture (output shapes, forward pass)
│   └── test_diffusion.py          # DDIM sampling shapes
├── test_agents/
│   ├── test_mof_master.py         # MOFMaster with mocked OpenAI API
│   ├── test_linkergen.py          # LinkerGen with mocked LangChain
│   └── test_qforge.py             # MFOModeller with mocked zeo++/MACE
└── test_mp_data/
    └── test_mp_extraction.py      # MPContribs script with mocked API
```

### What Gets Tested
- **Unit tests:** Model architecture shapes, data conversion logic, dataclass construction, config parsing — pure, no external deps
- **Mocked integration tests:** Agent workflows with mocked OpenAI/LangChain responses — verifies logic without API keys
- **Import smoke tests:** Every module imports without crashing

### Test Runner
- `pytest` added to pixi dependencies
- `pixi run test` task

## Section 3: GitHub Actions CI

### Workflow: `.github/workflows/ci.yml`

**Triggers:** push to main, pull requests

**Jobs:**

1. `smoke-test` (matrix: ubuntu-latest, macos-latest, windows-latest)
   - Install pixi
   - `pixi install` (default CPU env)
   - `pixi run test-imports` (verify all modules import)
   - `pixi run test` (run pytest suite)

2. `smoke-test-cuda` (ubuntu-latest only)
   - Install pixi
   - `pixi install -e cuda` (verify CUDA env resolves, no GPU needed)

## Section 4: Materials Project Data Extraction Script

### File: `scripts/mp_data_extraction.py`

### Two Modes

1. **Full download** — Pull entire MOFGen_2025 dataset from MPContribs, save as CSV + JSON
2. **Query interface** — Filter by properties (metal type, surface area range, pore size)

### Usage
```bash
export MP_API_KEY="your_key_here"

# Full download
pixi run mp-download --output data/mofgen_2025.csv

# Query subset
pixi run mp-query --metal Zr --min-surface-area 1000 --output data/zr_mofs.csv
```

### Implementation
- Uses `mp-api` and `mpcontribs-client` to connect to MPContribs project `MOFGen_2025`
- API key from `MP_API_KEY` env var or `.env` file (via `python-dotenv`)
- Outputs: CSV and JSON formats
- Clear error messages for missing/invalid API key

### Tests
- Mocked API responses to verify parsing/filtering logic

## Section 5: README Overhaul

### Updates
1. **Clear CPU vs CUDA installation** — `pixi install` (CPU) vs `pixi install -e cuda` (GPU)
2. **Troubleshooting section** — Common issues (TensorFlow on macOS ARM, CUDA not found)
3. **Materials Project data section** — API key setup, full download, query examples
4. **Testing section** — `pixi run test`
5. **Prerequisites** — What users need before starting (pixi, optionally GPU drivers, optionally API keys)
