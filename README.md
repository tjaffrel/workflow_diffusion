# Workflow Diffusion - MOF Generation and DFT Analysis

This repository contains tools for Metal-Organic Framework (MOF) generation using diffusion models and DFT analysis workflows.

## Environment Setup

This project uses **Pixi** for dependency management and environment setup.

### Install Pixi

**Linux/macOS:**
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows (PowerShell):**
```powershell
iwr https://pixi.sh/install.ps1 -useb | iex
```

Alternatively, see [pixi installation instructions](https://pixi.sh/dev/installation/) for other installation methods.

### Setup Environment
```bash
pixi install  # Install all dependencies
```

### Verify Installation
```bash
pixi run check-cuda  # Check PyTorch CUDA setup
pixi run check-tf    # Check TensorFlow installation
```

## Directory Structure

### `diffuse_materials/` - MOF Generation
Diffusion model for generating Metal-Organic Framework structures.

**Scripts:**
- `cif_to_tfrecord.py` - Convert CIF files to TFRecord format
- `train.py` - Train the diffusion model
- `dataset.py` - Data loading and preprocessing
- `model.py` - DiT (Diffusion Transformer) architecture
- `diffusion.py` - DDIM sampling for generation
- `vae.py` - VAE component (placeholder)

**Quick Start:**
```bash
# Convert CIF files to TFRecord
pixi run python diffuse_materials/cif_to_tfrecord.py --cif_dir diffuse_materials/qmof_subset --output mof_data.tfrecord

# Train model
pixi run python diffuse_materials/train.py --dataset_dir="mof_data.tfrecord" --batch_size=2 --max_train_steps=1000
```

**See `diffuse_materials/README.md` for detailed usage.**

### `dft_analysis/` - DFT Analysis
Jupyter notebooks for analyzing DFT calculation results and workflow completion.

**Notebooks:**
- `get_e_form.ipynb` - Calculate formation energies from DFT results
- `parse_workflow_steps.ipynb` - Analyze workflow completion statistics

**Usage:**
```bash
# Start Jupyter
pixi run jupyter lab

# Or run specific analysis
pixi run jupyter nbconvert --execute dft_analysis/get_e_form.ipynb
```

### `agents/` - MOF Agents
AI agents for MOF generation and analysis workflows.

**Agent 4 QForge:**
- `agents/agent_4_qforge/` - MOF Analysis & Optimization agent
- Replaces `run_jobs.ipynb` with clean Python workflow
- Combines zeo++ pore analysis and MACE force field relaxation
- Requires zeo++ software (set `ZEO_PATH` environment variable)

**Usage:**
```bash
# Example MOF generation
pixi run python example_mof_generation.py

# QForge MOF analysis (requires zeo++ installation)
pixi run python agents/agent_4_qforge/example_usage.py
```

**See `agents/agent_4_qforge/README.md` for detailed setup and configuration.**

## Available Tasks

```bash
# MOF Generation
pixi run convert --cif_dir /path/to/cifs --output /path/to/output.tfrecord
pixi run train --dataset_dir="mof_data.tfrecord"

# MOF Agent Example
pixi run python example_mof_generation.py
```

## Dependencies

**Core ML/AI:**
- PyTorch 2.5.1 (CUDA 12.1)
- TensorFlow 2.18.0
- CUDA support for RTX 

**Scientific Computing:**
- NumPy, SciPy
- PyMatGen
- Matplotlib, Plotly

**DFT Analysis:**
- python-dotenv
- seaborn
- pymongo
- mp-api (Materials Project API)

**Development:**
- ember-ai
- Jupyter Lab
- Fire (CLI framework)

**MOF Agents:**
- openai (OpenAI API client)

**Agent 4 QForge (MOF Analysis):**
- jobflow (workflow management)
- atomate2[forcefields] (MACE force field)
- fireworks (quantum execution manager)

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
