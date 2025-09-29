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

## Components

### 1. `cif_to_tfrecord.py` - Data Converter
Converts CIF (Crystallographic Information File) files to TFRecord format for training.

**Usage:**
```bash
pixi run python diffuse_materials/cif_to_tfrecord.py --cif_dir /path/to/cif/files --output /path/to/output.tfrecord
```

**Parameters:**
- `--cif_dir`: Directory containing .cif files
- `--output`: Output TFRecord file path
- `--max_structures`: Maximum number of structures to process (optional)

**Example:**
```bash
pixi run python diffuse_materials/cif_to_tfrecord.py --cif_dir diffuse_materials/qmof_subset --output mof_data.tfrecord
```

**Output TFRecord contains:**
- `frac_coords`: Fractional atomic coordinates [N_atoms × 3]
- `atom_types`: Atomic numbers [N_atoms]
- `lengths`: Unit cell lengths [3]
- `angles`: Unit cell angles [3] 
- `formula`: Chemical formula string

### 2. `dataset.py` - Data Loader
Loads MOF data from TFRecord files and converts to PyTorch tensors.

**Key Functions:**
- `MOFDataset`: PyTorch dataset class for TFRecord files
- `collate_fn`: Batch collation with padding to 256 atoms
- Preprocessing: Normalizes coordinates and atom types

**Usage:**
```python
from dataset import MOFDataset
dataset = MOFDataset(
    name="train",
    video_shape=[256, 1, 1, 4],  # [max_atoms, channels, height, width]
    dataset_paths=["mof_data.tfrecord"]
)
```

### 3. `model.py` - DiT Architecture
Diffusion Transformer model with spatial and temporal attention mechanisms.

**Key Classes:**
- `DiT`: Main model with configurable dimensions, layers, and heads
- `SpatialAttention`: Attention over atomic positions
- `TemporalAttention`: Attention over diffusion timesteps

**Model Parameters:**
- `in_channels`: Input channels (4 for [x,y,z,atom_type])
- `model_dim`: Hidden dimension (default: 1024)
- `layers`: Number of transformer layers (default: 12)
- `heads`: Number of attention heads (default: 16)

**Usage:**
```python
from model import DiT
model = DiT(
    in_channels=4,
    model_dim=1024,
    layers=12,
    heads=16
)
```

### 4. `diffusion.py` - DDIM Sampling
Implements DDIM (Denoising Diffusion Implicit Models) sampling with v-parameterization.

**Key Functions:**
- `DDIMSampler`: Main sampling class
- `generate()`: Generate new MOF structures
- `sample()`: Single sampling step

**Usage:**
```python
from diffusion import DDIMSampler
sampler = DDIMSampler(model, vae)
generated_mofs = sampler.generate(
    batch_size=4,
    num_steps=50,
    eta=0.0
)
```

### 5. `vae.py` - VAE Component
Placeholder VAE implementation (identity functions for now).

**Key Functions:**
- `encode()`: Encode input to latent space
- `decode()`: Decode latent to output space

### 6. `train.py` - Training Pipeline
Complete training pipeline with distributed support and checkpointing.

**Usage:**
```bash
pixi run python diffuse_materials/train.py --dataset_dir="mof_data.tfrecord" [options]
```

**Key Parameters:**
- `--dataset_dir`: Path to TFRecord file
- `--batch_size`: Batch size (default: 4)
- `--max_train_steps`: Training steps (default: 500000)
- `--validate_every`: Validation frequency (default: 20000)
- `--log_every`: Logging frequency (default: 1000)
- `--model_dim`: Model hidden dimension (default: 1024)
- `--layers`: Number of transformer layers (default: 12)
- `--heads`: Number of attention heads (default: 16)
- `--learning_rate`: Learning rate (default: 8e-5)

**Example Training:**
```bash
# Quick test training
pixi run python diffuse_materials/train.py \
    --dataset_dir="mof_data.tfrecord" \
    --batch_size=2 \
    --max_train_steps=1000 \
    --validate_every=500 \
    --log_every=50 \
    --model_dim=512 \
    --layers=4 \
    --heads=8

# Full training
pixi run python diffuse_materials/train.py \
    --dataset_dir="mof_data.tfrecord" \
    --batch_size=4 \
    --max_train_steps=500000 \
    --validate_every=20000 \
    --log_every=1000
```

## Complete Workflow

### 1. Setup Environment
```bash
pixi install  # Install all dependencies
```

### 2. Convert CIF to TFRecord
```bash
pixi run python diffuse_materials/cif_to_tfrecord.py \
    --cif_dir diffuse_materials/qmof_subset \
    --output mof_data.tfrecord
```

### 3. Train Model
```bash
pixi run python diffuse_materials/train.py \
    --dataset_dir="mof_data.tfrecord" \
    --batch_size=4 \
    --max_train_steps=500000
```

### 4. Generate New MOFs
```python
import torch
from model import DiT
from diffusion import DDIMSampler
from vae import VAE

# Load trained model
model = DiT(in_channels=4, model_dim=1024, layers=12, heads=16)
model.load_state_dict(torch.load("checkpoints/model_500000.pt"))
vae = VAE()

# Generate new structures
sampler = DDIMSampler(model, vae)
new_mofs = sampler.generate(batch_size=4, num_steps=50)
```

## Hardware Requirements
- CUDA-compatible GPU (required)
- Minimum 4GB VRAM for batch_size=4
- Multi-GPU setup for distributed training (optional)

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