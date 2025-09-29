# MOFGen CrystalGen - Diffusion Model for MOF Generation

This implementation generates Metal-Organic Framework (MOF) structures using diffusion models. MOFs contain hundreds to thousands of atoms (vs. ~12 in inorganic crystals), requiring novel approaches beyond traditional crystal generation models.

## Approach

**Point Cloud Representation**: Each MOF with A atoms is represented as [A, 4] matrix:
- Columns 0-2: Normalized atomic coordinates (x, y, z)
- Column 3: Normalized atomic number (element type)
- Maximum atoms: A = 256

**Model Architecture**: 
- Diffusion Transformer (DiT) with spatial/temporal attention
- Order-invariant processing (no positional encoding)
- Resolution-preserving (no downsampling/upsampling)

## Files

- **`dataset.py`**: Loads MOF data from TFRecord files → tensor format
- **`model.py`**: DiT architecture with dual attention mechanisms  
- **`diffusion.py`**: DDIM sampling with v-parameterization
- **`train.py`**: Training pipeline with distributed support

## Setup Requirements

### Hardware Requirements
- CUDA-compatible GPU (required)
- Minimum 4GB VRAM for batch_size=4
- Multi-GPU setup for distributed training (optional)

### Software Installation
**Option 1: Pixi (Recommended)**
```bash
pixi install  # Creates virtual environment with all dependencies
pixi run activate  # Activate environment
```

**Option 2: Manual pip**
```bash
pip install torch torchvision torchaudio tensorflow einops absl-py fire matplotlib imageio tqdm pymatgen numpy scipy
```

### Data Preparation
**You need:** TFRecord files with MOF structures

**To create TFRecord files from CIF files:**
```bash
python cif_to_tfrecord.py --cif_dir=/path/to/cif/files --output=/path/to/output.tfrecord
```

**TFRecord contains:**
- `frac_coords`: Fractional atomic coordinates [N_atoms × 3]
- `atom_types`: Atomic numbers [N_atoms]
- `lengths`: Unit cell lengths [3]
- `angles`: Unit cell angles [3] 
- `formula`: Chemical formula string

### Quick Start
1. Setup environment: `pixi install`
2. Prepare TFRecord data: `pixi run python diffuse_materials/cif_to_tfrecord.py --cif_dir diffuse_materials/qmof_subset --output mof_data.tfrecord`
3. Run training: `pixi run python diffuse_materials/train.py --dataset_dir="mof_data.tfrecord"`
4. Generate MOFs: Use trained model with `diffusion.generate()`

**Example:** Processing 1000 MOF structures
```bash
python cif_to_tfrecord.py --cif_dir=/data/cif_files --output=/data/train.tfrecord --max_structures=1000
```

**Model Training Setup:**
- Input: MOF structures (up to 256 atoms)
- Batch size: 4 (default)
- Learning rate: 8e-5 (default)
- Steps: 500k (default)
- Validation: Every 20k steps

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