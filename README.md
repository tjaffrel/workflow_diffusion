# MOFGen — AI-Driven Metal-Organic Framework Generation

Tools for Metal-Organic Framework (MOF) generation using diffusion models and AI agents.

[![CI](https://github.com/tjaffrel/mofgen/actions/workflows/ci.yml/badge.svg)](https://github.com/tjaffrel/mofgen/actions/workflows/ci.yml)

## Prerequisites

- [Pixi](https://pixi.sh) package manager

Optional (depending on features used):
- NVIDIA GPU + drivers (for CUDA-accelerated training)
- OpenAI API key (for LLM-based agents: MOFMaster, LinkerGen)
- Materials Project API key (for data extraction)

## Installation

### 1. Install Pixi

**Linux / macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows (PowerShell):**
```powershell
iwr https://pixi.sh/install.ps1 -useb | iex
```

### 2. Install Dependencies

**CPU only (works on all platforms):**
```bash
pixi install
```

**With CUDA support (Linux/Windows with NVIDIA GPU):**
```bash
pixi install -e cuda
```

### 3. Verify Installation

```bash
pixi run test-imports   # Verify all modules load
pixi run check-tf       # Check TensorFlow
pixi run check-cuda     # Check PyTorch CUDA (GPU env only)
```

## Quick Start

### Diffusion Model (MOF Structure Generation)

```bash
# Convert CIF files to TFRecord format
pixi run python diffuse_materials/cif_to_tfrecord.py \
    --cif_dir /path/to/cifs --output mof_data.tfrecord

# Train the diffusion model (requires GPU)
pixi run train --dataset_dir mof_data.tfrecord
```

See [`diffuse_materials/README.md`](diffuse_materials/README.md) for details.

### AI Agents

```bash
# MOFMaster — LLM-based MOF generation (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your_key"
pixi run python example_mof_generation.py

# LinkerGen — MOF linker generation
pixi run python agents/agent_2_linkergen/example_usage.py

# QForge — MOF analysis with zeo++ and MACE (requires zeo++)
pixi run python agents/agent_4_qforge/example_usage.py
```

See [`agents/README.md`](agents/README.md) for details.

### Working with the Data

The MOFGen_2025 DFT relaxation trajectories are publicly available on S3
(`materialsproject-contribs` bucket). Each trajectory contains per-frame
energies (eV/atom), forces (eV/A), and stresses (eV/A^3) across all ionic
steps. No API key is required.

```bash
# Extract trajectories (downloads from public S3, no credentials needed)
pixi run extract-trajectories

# Process more trajectories at once
pixi run python examples/extract_trajectories.py --max-trajectories 10
```

The script uses the [`emmet-core`](https://github.com/materialsproject/emmet)
`RelaxTrajectory` model and supports conversion to pymatgen and ASE trajectory
objects. See [`examples/extract_trajectories.py`](examples/extract_trajectories.py)
for details.

### Materials Project Data Extraction

Download and query the MOFGen_2025 dataset from [MPContribs](https://next-gen.materialsproject.org/contribs/projects/MOFGen_2025).

```bash
# Set your Materials Project API key
# Get one at: https://next-gen.materialsproject.org/api#api-key
export MP_API_KEY="your_key"

# Download full dataset
pixi run mp-download --output data/mofgen_2025.csv

# Query by metal
pixi run mp-query --metal Zr --output data/zr_mofs.csv

# Query by surface area range
pixi run mp-query --min-surface-area 1000 --output data/high_sa_mofs.csv
```

## Running Tests

```bash
pixi run test
```

## Project Structure

```
mofgen/
├── diffuse_materials/     # Diffusion model pipeline (DiT architecture)
│   ├── model.py           # DiT transformer
│   ├── diffusion.py       # DDIM sampling
│   ├── dataset.py         # TFRecord data loader
│   ├── cif_to_tfrecord.py # CIF → TFRecord converter
│   ├── train.py           # Training script
│   └── vae.py             # VAE placeholder
├── agents/                # LLM agent framework
│   ├── mof_master.py      # Primary MOF generation agent
│   ├── agent_2_linkergen/ # Linker generation agent
│   └── agent_4_qforge/    # MOF analysis & optimization
├── examples/              # User-facing example scripts
│   └── extract_trajectories.py  # Extract forces/stresses from MOFGen_2025
├── scripts/               # Utility scripts
│   └── mp_data_extraction.py    # Materials Project data download/query
├── dft_analysis/          # DFT analysis and plotting (figure reproduction)
├── tests/                 # Test suite
├── pixi.toml              # Environment & dependency config
├── LICENSE                # MIT License
└── README.md
```

## Troubleshooting

### `pixi install` fails with "Unexpected keys"

Make sure you are using pixi >= 0.30. Update with:
```bash
pixi self-update
```

### TensorFlow fails to install on macOS ARM (M1/M2/M3)

TensorFlow from conda-forge should work on osx-arm64. If resolution fails:
```bash
# Try clearing the pixi cache
pixi clean
pixi install
```

### CUDA not detected

Verify your NVIDIA drivers are installed:
```bash
nvidia-smi
```

Make sure you installed the CUDA environment:
```bash
pixi install -e cuda
pixi run check-cuda
```

### OpenAI API errors in agents

The MOFMaster and LinkerGen agents require an OpenAI API key:
```bash
export OPENAI_API_KEY="your_key"
```

## Citation

```bibtex
@misc{inizan2025agenticaidiscoverymetalorganic,
      title={System of Agentic AI for the Discovery of Metal-Organic Frameworks},
      author={Theo Jaffrelot Inizan and Sherry Yang and Aaron Kaplan and Yen-hsu Lin and Jian Yin and Saber Mirzaei and Mona Abdelgaid and Ali H. Alawadhi and KwangHwan Cho and Zhiling Zheng and Ekin Dogus Cubuk and Christian Borgs and Jennifer T. Chayes and Kristin A. Persson and Omar M. Yaghi},
      year={2025},
      eprint={2504.14110},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```
