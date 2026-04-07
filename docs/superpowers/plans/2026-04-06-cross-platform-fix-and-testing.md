# Cross-Platform Fix, Test Suite, CI, and MP Data Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MOFGen installable and testable on any platform (Linux, macOS, Windows) with proper CI, and add a Materials Project data extraction script.

**Architecture:** Fix pixi.toml with CPU/CUDA feature split, clean up conflicting pyproject.toml pixi sections, add pytest-based test suite with mocked external dependencies, GitHub Actions CI matrix across 3 OS, and a `scripts/mp_data_extraction.py` for MPContribs data access.

**Tech Stack:** pixi, pytest, unittest.mock, GitHub Actions, mp-api, mpcontribs-client, python-dotenv

---

### Task 1: Fix pixi.toml — CPU/CUDA Feature Split

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Rewrite pixi.toml with feature-based environments**

Replace the entire contents of `pixi.toml` with:

```toml
[project]
name = "mofgen"
description = "MOF Generation workflow with diffusion models and AI agents"
version = "0.1.0"
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[dependencies]
python = "3.11.*"
numpy = "*"
scipy = "*"
pymatgen = "*"
matplotlib = "*"
imageio = "*"
tqdm = "*"
einops = "*"
absl-py = "*"
fire = "*"
pandas = "*"
pytest = "*"
pytest-mock = "*"
cpuonly = {version = "*", channel = "pytorch"}
pytorch = ">=2.5.1,<2.6"
torchvision = ">=0.20.1,<0.21"
torchaudio = ">=2.5.1,<2.6"
tensorflow = ">=2.15,<3"

[pypi-dependencies]
python-dotenv = "*"
seaborn = "*"
pymongo = "*"
mp-api = "*"
mpcontribs-client = "*"
openai = "*"
jobflow = "*"
"atomate2[forcefields]" = "*"
fireworks = "*"
langchain = "*"
langchain-openai = "*"
ember-ai = "*"

[feature.cuda]
platforms = ["linux-64", "win-64"]

[feature.cuda.dependencies]
cuda-version = "12.1.*"
pytorch = {version = ">=2.5.1,<2.6", channel = "pytorch"}
torchvision = {version = ">=0.20.1,<0.21", channel = "pytorch"}
torchaudio = {version = ">=2.5.1,<2.6", channel = "pytorch"}

[environments]
default = {solve-group = "default"}
cuda = {features = ["cuda"], solve-group = "cuda"}

[tasks]
test = "pytest tests/ -v"
test-imports = "python -c \"import agents; import diffuse_materials.model; import diffuse_materials.diffusion; import diffuse_materials.vae; import diffuse_materials.dataset; import diffuse_materials.cif_to_tfrecord; print('All imports OK')\""
train = "python diffuse_materials/train.py --dataset_dir sample_data"
convert = "python diffuse_materials/cif_to_tfrecord.py --cif_dir /path/to/cifs --output /path/to/output.tfrecord"
check-cuda = "python -c \"import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')\""
check-tf = "python -c \"import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')\""
mp-download = "python scripts/mp_data_extraction.py download"
mp-query = "python scripts/mp_data_extraction.py query"
```

- [ ] **Step 2: Verify pixi.toml is valid syntax**

Run: `cd /home/theoj/project/mofgen && pixi info`
Expected: No syntax errors. Shows project info with `default` and `cuda` environments listed.

- [ ] **Step 3: Commit**

```bash
git add pixi.toml
git commit -m "fix: rewrite pixi.toml with CPU/CUDA feature split

Default env is CPU-only and works on all platforms.
CUDA env is opt-in via 'pixi install -e cuda' for Linux/Windows with GPU."
```

---

### Task 2: Clean Up pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Remove broken pixi sections from pyproject.toml**

Replace the entire contents of `pyproject.toml` with:

```toml
[project]
name = "mofgen"
description = "MOF Generation workflow with diffusion models and AI agents"
version = "0.1.0"
authors = [
    {name = "MOFGen Team"}
]
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    "tensorflow>=2.15.0",
    "einops>=0.6.0",
    "absl-py>=1.4.0",
    "fire>=0.5.0",
    "matplotlib>=3.7.0",
    "imageio>=2.25.0",
    "tqdm>=4.65.0",
    "pymatgen>=2023.12.18",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "fix: remove broken pixi sections from pyproject.toml

pixi.toml is the single source of truth for pixi config.
pyproject.toml now only has standard Python project metadata."
```

---

### Task 3: Add diffuse_materials __init__.py for importability

**Files:**
- Create: `diffuse_materials/__init__.py`

- [ ] **Step 1: Create the init file**

```python
"""Diffusion model pipeline for MOF structure generation."""
```

- [ ] **Step 2: Fix relative imports in train.py**

In `diffuse_materials/train.py`, the imports are bare module names (`from model import DiT`), which only work when running from inside the directory. Change lines 17-20 from:

```python
from model import DiT
from vae import VAE
from diffusion import Diffusion
from dataset import MOFDataset
```

to:

```python
from diffuse_materials.model import DiT
from diffuse_materials.vae import VAE
from diffuse_materials.diffusion import Diffusion
from diffuse_materials.dataset import MOFDataset
```

- [ ] **Step 3: Verify imports work**

Run: `cd /home/theoj/project/mofgen && python -c "import diffuse_materials.model; import diffuse_materials.diffusion; import diffuse_materials.vae; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add diffuse_materials/__init__.py diffuse_materials/train.py
git commit -m "fix: make diffuse_materials importable as a package

Add __init__.py and convert bare imports to absolute imports in train.py."
```

---

### Task 4: Add Test Fixtures (conftest.py)

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_diffusion/__init__.py`
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_mp_data/__init__.py`

- [ ] **Step 1: Create test directory structure and conftest.py**

Create `tests/__init__.py` (empty file).

Create `tests/test_diffusion/__init__.py` (empty file).

Create `tests/test_agents/__init__.py` (empty file).

Create `tests/test_mp_data/__init__.py` (empty file).

Create `tests/conftest.py`:

```python
"""Shared test fixtures for MOFGen test suite."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_cif_content():
    """Return a minimal valid CIF string for testing."""
    return """data_test_mof
_cell_length_a   10.0
_cell_length_b   10.0
_cell_length_c   10.0
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number   1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zn1 Zn 0.0 0.0 0.0
C1 C 0.25 0.25 0.25
O1 O 0.5 0.5 0.5
"""


@pytest.fixture
def sample_cif_file(tmp_dir, sample_cif_content):
    """Write sample CIF content to a temp file and return the path."""
    cif_path = tmp_dir / "test_mof.cif"
    cif_path.write_text(sample_cif_content)
    return cif_path


@pytest.fixture
def mock_openai_response():
    """Return a mock OpenAI chat completion response structure."""
    class MockMessage:
        content = (
            "Structure 1:\n"
            "CIF: data_gen\n_cell_length_a 10.0\n"
            "Formula: ZnC4H4O4\n"
            "Properties: Stable MOF with moderate porosity"
        )

    class MockChoice:
        message = MockMessage()

    class MockCompletion:
        choices = [MockChoice()]

    return MockCompletion()
```

- [ ] **Step 2: Commit**

```bash
git add tests/
git commit -m "test: add test directory structure and shared fixtures"
```

---

### Task 5: Test DiT Model Architecture

**Files:**
- Create: `tests/test_diffusion/test_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diffusion/test_model.py`:

```python
"""Tests for DiT model architecture — verifies shapes and forward pass."""

import pytest
import torch

from diffuse_materials.model import DiT, AttentionType, RotaryType


@pytest.fixture
def small_dit():
    """A small DiT for fast testing (2 layers, 64 dim)."""
    return DiT(
        in_channels=4,
        patch_size=2,
        dim=64,
        num_layers=2,
        num_heads=4,
        action_dim=10,
        max_frames=4,
    )


class TestDiTShapes:
    def test_forward_output_shape(self, small_dit):
        B, T, H, W, C = 2, 4, 8, 8, 4
        x = torch.randn(B, T, H, W, C)
        t = torch.randint(0, 1000, (B, T))
        actions = torch.randn(B, T, 10)
        out = small_dit(x, t, actions)
        assert out.shape == (B, T, H, W, C)

    def test_patchify_shape(self, small_dit):
        B, T, H, W, C = 2, 4, 8, 8, 4
        x = torch.randn(B, T, H, W, C)
        patched = small_dit.patchify(x)
        assert patched.shape == (B, T, H // 2, W // 2, 64)

    def test_timestep_embedding_shape(self, small_dit):
        t = torch.tensor([0, 500, 999])
        emb = small_dit.timestep_embedding(t, dim=256)
        assert emb.shape == (3, 256)

    def test_different_batch_sizes(self, small_dit):
        for B in [1, 3]:
            T, H, W, C = 4, 8, 8, 4
            x = torch.randn(B, T, H, W, C)
            t = torch.randint(0, 1000, (B, T))
            actions = torch.randn(B, T, 10)
            out = small_dit(x, t, actions)
            assert out.shape == (B, T, H, W, C)


class TestAttentionTypes:
    def test_attention_type_enum(self):
        assert AttentionType.SPATIAL == "spatial"
        assert AttentionType.TEMPORAL == "temporal"

    def test_rotary_type_enum(self):
        assert RotaryType.STANDARD == "standard"
        assert RotaryType.PIXEL == "pixel"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_diffusion/test_model.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_diffusion/test_model.py
git commit -m "test: add DiT model architecture shape tests"
```

---

### Task 6: Test Diffusion Module

**Files:**
- Create: `tests/test_diffusion/test_diffusion.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_diffusion/test_diffusion.py`:

```python
"""Tests for DDIM diffusion — schedule, q_sample, loss shapes."""

import pytest
import torch

from diffuse_materials.diffusion import Diffusion


@pytest.fixture
def diffusion():
    return Diffusion(timesteps=100, sampling_timesteps=5)


class TestDiffusionSchedule:
    def test_alphas_cumprod_shape(self, diffusion):
        assert diffusion.alphas_cumprod.shape == (100,)

    def test_alphas_cumprod_decreasing(self, diffusion):
        ac = diffusion.alphas_cumprod
        assert (ac[:-1] >= ac[1:]).all()

    def test_alphas_cumprod_range(self, diffusion):
        ac = diffusion.alphas_cumprod
        assert ac.min() > 0
        assert ac.max() <= 1


class TestQSample:
    def test_q_sample_shape(self, diffusion):
        B, T, H, W, C = 2, 4, 8, 8, 4
        x = torch.randn(B, T, H, W, C)
        t = torch.randint(0, 100, (B, T))
        noise = torch.randn_like(x)
        noisy = diffusion.q_sample(x, t, noise)
        assert noisy.shape == x.shape

    def test_q_sample_at_t0_close_to_x(self, diffusion):
        B, T, H, W, C = 1, 1, 4, 4, 4
        x = torch.randn(B, T, H, W, C)
        t = torch.zeros(B, T, dtype=torch.long)
        noise = torch.randn_like(x)
        noisy = diffusion.q_sample(x, t, noise)
        # At t=0 alpha_cumprod is close to 1, so noisy should be close to x
        assert torch.allclose(noisy, x, atol=0.5)


class TestLossFn:
    def test_loss_scalar(self, diffusion):
        B, T, H, W, C = 2, 4, 8, 8, 4
        from diffuse_materials.model import DiT

        model = DiT(
            in_channels=4, patch_size=2, dim=64,
            num_layers=1, num_heads=4, action_dim=10, max_frames=4,
        )
        x = torch.randn(B, T, H, W, C)
        actions = torch.randn(B, T, 10)
        loss = diffusion.loss_fn(model, x, actions)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0


class TestSchedulingMatrix:
    def test_pyramid_shape(self, diffusion):
        horizon = 3
        mat = diffusion.generate_pyramid_scheduling_matrix(horizon)
        expected_height = diffusion.sampling_timesteps + horizon
        assert mat.shape == (expected_height, horizon)

    def test_pyramid_values_clipped(self, diffusion):
        mat = diffusion.generate_pyramid_scheduling_matrix(4)
        assert mat.min() >= 0
        assert mat.max() <= diffusion.sampling_timesteps
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_diffusion/test_diffusion.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_diffusion/test_diffusion.py
git commit -m "test: add diffusion schedule and sampling tests"
```

---

### Task 7: Test VAE Placeholder

**Files:**
- Create: `tests/test_diffusion/test_vae.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_diffusion/test_vae.py`:

```python
"""Tests for VAE placeholder — encode/decode are identity."""

import torch

from diffuse_materials.vae import VAE


def test_encode_identity():
    vae = VAE()
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(vae.encode(x), x)


def test_decode_identity():
    vae = VAE()
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(vae.decode(x), x)


def test_latent_channels():
    vae = VAE()
    assert vae.config.latent_channels == 4
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_diffusion/test_vae.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_diffusion/test_vae.py
git commit -m "test: add VAE placeholder tests"
```

---

### Task 8: Test CIF-to-TFRecord Conversion

**Files:**
- Create: `tests/test_diffusion/test_cif_to_tfrecord.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_diffusion/test_cif_to_tfrecord.py`:

```python
"""Tests for CIF to TFRecord conversion — parsing and serialization."""

import pytest
import numpy as np

from diffuse_materials.cif_to_tfrecord import parse_cif_file, create_tfrecord_example


class TestParseCif:
    def test_parse_returns_five_elements(self, sample_cif_file):
        result = parse_cif_file(str(sample_cif_file))
        assert len(result) == 5

    def test_frac_coords_shape(self, sample_cif_file):
        frac_coords, _, _, _, _ = parse_cif_file(str(sample_cif_file))
        # 3 atoms (Zn, C, O), each with 3 fractional coordinates
        assert frac_coords.shape == (3, 3)

    def test_atom_types(self, sample_cif_file):
        _, atom_types, _, _, _ = parse_cif_file(str(sample_cif_file))
        # Zn=30, C=6, O=8
        assert atom_types.numpy().tolist() == [30, 6, 8]

    def test_lengths(self, sample_cif_file):
        _, _, lengths, _, _ = parse_cif_file(str(sample_cif_file))
        np.testing.assert_allclose(lengths.numpy(), [10.0, 10.0, 10.0])

    def test_angles(self, sample_cif_file):
        _, _, _, angles, _ = parse_cif_file(str(sample_cif_file))
        np.testing.assert_allclose(angles.numpy(), [90.0, 90.0, 90.0])

    def test_formula(self, sample_cif_file):
        _, _, _, _, formula = parse_cif_file(str(sample_cif_file))
        # pymatgen reduced formula for Zn1 C1 O1
        assert isinstance(formula, str)
        assert len(formula) > 0


class TestCreateTfrecordExample:
    def test_creates_valid_example(self, sample_cif_file):
        frac_coords, atom_types, lengths, angles, formula = parse_cif_file(
            str(sample_cif_file)
        )
        example = create_tfrecord_example(
            frac_coords, atom_types, lengths, angles, formula
        )
        serialized = example.SerializeToString()
        assert len(serialized) > 0
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_diffusion/test_cif_to_tfrecord.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_diffusion/test_cif_to_tfrecord.py
git commit -m "test: add CIF-to-TFRecord conversion tests"
```

---

### Task 9: Test MOFMaster Agent (Mocked OpenAI)

**Files:**
- Create: `tests/test_agents/test_mof_master.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_agents/test_mof_master.py`:

```python
"""Tests for MOFMaster agent — mocked OpenAI, tests data flow and parsing."""

import pytest
from unittest.mock import patch, MagicMock

from agents.mof_master import (
    MOFMaster,
    MOFGenerationMode,
    MOFGenerationRequest,
    MOFGenerationResponse,
    MOFStructure,
)


@pytest.fixture
def mocked_master(mock_openai_response):
    """Create a MOFMaster with mocked OpenAI client."""
    with patch("agents.mof_master.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        MockOpenAI.return_value = mock_client
        master = MOFMaster(api_key="fake-key")
        yield master


class TestMOFStructureDataclass:
    def test_create_basic(self):
        s = MOFStructure(cif_content="data_test", formula="ZnO")
        assert s.cif_content == "data_test"
        assert s.formula == "ZnO"
        assert s.metal_sbu is None

    def test_create_with_metadata(self):
        s = MOFStructure(
            cif_content="data_test",
            formula="ZnO",
            metal_sbu="Zn",
            properties={"pore_size": 5.0},
            generation_metadata={"mode": "basic"},
        )
        assert s.metal_sbu == "Zn"
        assert s.properties["pore_size"] == 5.0


class TestMOFGenerationRequest:
    def test_basic_request(self):
        req = MOFGenerationRequest(mode=MOFGenerationMode.BASIC, count=3)
        assert req.count == 3
        assert req.metal is None

    def test_metal_request(self):
        req = MOFGenerationRequest(
            mode=MOFGenerationMode.METAL_SPECIFIC, count=1, metal="Zr"
        )
        assert req.metal == "Zr"


class TestMOFMasterInit:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                MOFMaster()


class TestMOFMasterGeneration:
    def test_generate_basic(self, mocked_master):
        response = mocked_master.generate_basic_structures(count=1)
        assert isinstance(response, MOFGenerationResponse)
        assert response.success_count == 1
        assert response.mode_used == MOFGenerationMode.BASIC
        assert len(response.structures) == 1

    def test_generate_metal_specific(self, mocked_master):
        response = mocked_master.generate_metal_specific_structures(
            metal="Zn", count=2
        )
        assert response.success_count == 2
        assert response.mode_used == MOFGenerationMode.METAL_SPECIFIC
        for s in response.structures:
            assert s.metal_sbu == "Zn"

    def test_generate_composition_specific(self, mocked_master):
        comp = {"Zn": 0.3, "C": 0.4, "O": 0.3}
        response = mocked_master.generate_composition_specific_structures(
            composition=comp, count=1
        )
        assert response.success_count == 1
        assert response.structures[0].composition == comp

    def test_generation_time_recorded(self, mocked_master):
        response = mocked_master.generate_basic_structures(count=1)
        assert response.generation_time >= 0

    def test_metadata_contains_model(self, mocked_master):
        response = mocked_master.generate_basic_structures(count=1)
        assert "model_used" in response.metadata


class TestParseStructureResponse:
    def test_parses_cif_from_response(self, mocked_master):
        text = (
            "Structure 1:\n"
            "CIF: data_gen\n"
            "_cell_length_a 10.0\n"
            "Formula: ZnC4H4O4\n"
            "Properties: Stable MOF"
        )
        s = mocked_master._parse_structure_response(text, 1, "basic")
        assert "data_gen" in s.cif_content
        assert s.formula == "ZnC4H4O4"

    def test_placeholder_on_empty_response(self, mocked_master):
        s = mocked_master._parse_structure_response("No CIF here", 1, "basic")
        # Should generate placeholder CIF
        assert "data_1" in s.cif_content
        assert s.formula == "ZnC4H4O4"  # placeholder formula
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_agents/test_mof_master.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agents/test_mof_master.py
git commit -m "test: add MOFMaster agent tests with mocked OpenAI"
```

---

### Task 10: Test LinkerGen Agent (Mocked LangChain)

**Files:**
- Create: `tests/test_agents/test_linkergen.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_agents/test_linkergen.py`:

```python
"""Tests for LinkerGenAgent — mocked LangChain, tests SMILES and formula generation."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agents.agent_2_linkergen.linkergen_agent import LinkerGenAgent, LinkerGenConfig


@pytest.fixture
def mock_llm_response():
    """Mock LangChain LLM response."""
    mock_resp = MagicMock()
    mock_resp.content = "1. CC(=O)O\n2. c1ccccc1\n3. OC(=O)c1ccc(C(=O)O)cc1"
    return mock_resp


@pytest.fixture
def agent_with_mock(mock_llm_response):
    """Create LinkerGenAgent with mocked LLM."""
    with patch(
        "agents.agent_2_linkergen.linkergen_agent.ChatOpenAI"
    ) as MockLLM:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response
        MockLLM.return_value = mock_instance
        config = LinkerGenConfig(openai_api_key="fake-key")
        agent = LinkerGenAgent(config)
        yield agent


@pytest.fixture
def examples_file(tmp_dir):
    """Create a temporary examples file."""
    f = tmp_dir / "examples.txt"
    f.write_text("OC(=O)c1ccc(C(=O)O)cc1\nOC(=O)c1cccc(C(=O)O)c1")
    return str(f)


class TestLinkerGenConfig:
    def test_defaults(self):
        cfg = LinkerGenConfig()
        assert cfg.model_name == "gpt-4"
        assert cfg.temperature == 1.0


class TestLoadExamples:
    def test_load_existing_file(self, agent_with_mock, examples_file):
        content = agent_with_mock._load_examples(examples_file)
        assert "OC(=O)" in content

    def test_load_missing_file(self, agent_with_mock):
        result = agent_with_mock._load_examples("/nonexistent/path.txt")
        assert result is None


class TestGenerateSmiles:
    def test_returns_string(self, agent_with_mock, examples_file):
        result = agent_with_mock.generate_smiles_from_smiles(
            examples_file, num_linkers=3
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_saves_to_output_file(self, agent_with_mock, examples_file, tmp_dir):
        out = str(tmp_dir / "output.txt")
        agent_with_mock.generate_smiles_from_smiles(
            examples_file, num_linkers=3, output_file=out
        )
        assert Path(out).exists()
        assert len(Path(out).read_text()) > 0

    def test_raises_on_missing_examples(self, agent_with_mock):
        with pytest.raises(ValueError, match="Could not load"):
            agent_with_mock.generate_smiles_from_smiles("/no/file.txt")


class TestGenerateFormula:
    def test_returns_string(self, agent_with_mock, examples_file):
        result = agent_with_mock.generate_formula_from_formula(
            examples_file, num_linkers=5
        )
        assert isinstance(result, str)
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_agents/test_linkergen.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agents/test_linkergen.py
git commit -m "test: add LinkerGen agent tests with mocked LangChain"
```

---

### Task 11: Test MFOModeller Agent (Mocked Dependencies)

**Files:**
- Create: `tests/test_agents/test_qforge.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_agents/test_qforge.py`:

```python
"""Tests for MFOModeller — config, structure loading, batch validation."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agents.agent_4_qforge.mof_modeller import MFOModeller, MFOModellerConfig


class TestMFOModellerConfig:
    def test_defaults(self):
        cfg = MFOModellerConfig()
        assert cfg.zeopp_path is None
        assert cfg.zeopp_nproc == 3
        assert cfg.sorbates == ["N2", "CO2", "H2O"]
        assert cfg.run_local is True

    def test_custom_config(self):
        cfg = MFOModellerConfig(
            zeopp_path="/usr/local/bin/zeo++",
            zeopp_nproc=8,
            sorbates=["Ar"],
        )
        assert cfg.zeopp_path == "/usr/local/bin/zeo++"
        assert cfg.sorbates == ["Ar"]


class TestMFOModellerInit:
    def test_default_init(self):
        m = MFOModeller()
        assert m.config.run_local is True

    def test_custom_init(self):
        cfg = MFOModellerConfig(store_results=False)
        m = MFOModeller(config=cfg)
        assert m.config.store_results is False


class TestAnalyzeStructure:
    def test_raises_on_invalid_cif(self, tmp_dir):
        bad_cif = tmp_dir / "bad.cif"
        bad_cif.write_text("not a cif file")
        m = MFOModeller()
        with pytest.raises(ValueError, match="Could not parse CIF"):
            m.analyze_structure(bad_cif)


class TestBatchAnalyze:
    def test_raises_on_empty_dir(self, tmp_dir):
        m = MFOModeller()
        with pytest.raises(ValueError, match="No CIF files found"):
            m.batch_analyze(tmp_dir)
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_agents/test_qforge.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agents/test_qforge.py
git commit -m "test: add MFOModeller agent tests"
```

---

### Task 12: Create Materials Project Data Extraction Script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/mp_data_extraction.py`

- [ ] **Step 0: Create scripts package init**

Create `scripts/__init__.py` (empty file) so tests can import from `scripts.mp_data_extraction`.

- [ ] **Step 1: Write the extraction script**

Create `scripts/mp_data_extraction.py`:

```python
"""Materials Project data extraction for MOFGen_2025.

Downloads and queries the MOFGen_2025 dataset from MPContribs.

Usage:
    # Full download
    python scripts/mp_data_extraction.py download --output data/mofgen_2025.csv

    # Query subset
    python scripts/mp_data_extraction.py query --metal Zr --output data/zr_mofs.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "MOFGen_2025"


def get_api_key():
    """Get MP API key from environment."""
    key = os.getenv("MP_API_KEY")
    if not key:
        print(
            "Error: MP_API_KEY not set.\n\n"
            "Set it via environment variable:\n"
            "  export MP_API_KEY='your_key_here'\n\n"
            "Or add it to a .env file in the project root:\n"
            "  MP_API_KEY=your_key_here\n\n"
            "Get your API key at: https://next-gen.materialsproject.org/api#api-key"
        )
        sys.exit(1)
    return key


def get_client(api_key):
    """Create MPContribs client."""
    from mpcontribs.client import Client

    return Client(api_key)


def download(output, format):
    """Download the full MOFGen_2025 dataset."""
    api_key = get_api_key()
    client = get_client(api_key)

    print(f"Fetching contributions from project '{PROJECT_NAME}'...")
    contributions = client.get_contributions(PROJECT_NAME)
    df = pd.json_normalize(contributions)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        # Save both
        csv_path = output_path.with_suffix(".csv")
        json_path = output_path.with_suffix(".json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved {len(df)} entries to {csv_path} and {json_path}")
        return df

    print(f"Saved {len(df)} entries to {output_path}")
    return df


def query(metal=None, min_surface_area=None, max_surface_area=None,
          min_pore_size=None, max_pore_size=None, formula=None,
          output="query_results.csv", format="csv"):
    """Query the MOFGen_2025 dataset with filters."""
    api_key = get_api_key()
    client = get_client(api_key)

    print(f"Querying project '{PROJECT_NAME}' with filters...")
    contributions = client.get_contributions(PROJECT_NAME)
    df = pd.json_normalize(contributions)

    # Apply filters
    if metal:
        metal_col = [c for c in df.columns if "metal" in c.lower()]
        if metal_col:
            df = df[df[metal_col[0]].str.contains(metal, case=False, na=False)]
        else:
            # Fall back to filtering on formula or identifier columns
            formula_col = [c for c in df.columns if "formula" in c.lower() or "identifier" in c.lower()]
            if formula_col:
                df = df[df[formula_col[0]].str.contains(metal, case=False, na=False)]

    if formula:
        formula_col = [c for c in df.columns if "formula" in c.lower()]
        if formula_col:
            df = df[df[formula_col[0]].str.contains(formula, case=False, na=False)]

    if min_surface_area is not None:
        sa_col = [c for c in df.columns if "surface" in c.lower() and "area" in c.lower()]
        if sa_col:
            df = df[pd.to_numeric(df[sa_col[0]], errors="coerce") >= min_surface_area]

    if max_surface_area is not None:
        sa_col = [c for c in df.columns if "surface" in c.lower() and "area" in c.lower()]
        if sa_col:
            df = df[pd.to_numeric(df[sa_col[0]], errors="coerce") <= max_surface_area]

    if min_pore_size is not None:
        pore_col = [c for c in df.columns if "pore" in c.lower() and ("size" in c.lower() or "diameter" in c.lower())]
        if pore_col:
            df = df[pd.to_numeric(df[pore_col[0]], errors="coerce") >= min_pore_size]

    if max_pore_size is not None:
        pore_col = [c for c in df.columns if "pore" in c.lower() and ("size" in c.lower() or "diameter" in c.lower())]
        if pore_col:
            df = df[pd.to_numeric(df[pore_col[0]], errors="coerce") <= max_pore_size]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        csv_path = output_path.with_suffix(".csv")
        json_path = output_path.with_suffix(".json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved {len(df)} filtered entries to {csv_path} and {json_path}")
        return df

    print(f"Saved {len(df)} filtered entries to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract MOFGen_2025 data from Materials Project / MPContribs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download subcommand
    dl_parser = subparsers.add_parser("download", help="Download full dataset")
    dl_parser.add_argument(
        "--output", default="data/mofgen_2025.csv",
        help="Output file path (default: data/mofgen_2025.csv)"
    )
    dl_parser.add_argument(
        "--format", choices=["csv", "json", "both"], default="both",
        help="Output format (default: both)"
    )

    # Query subcommand
    q_parser = subparsers.add_parser("query", help="Query dataset with filters")
    q_parser.add_argument("--metal", help="Filter by metal element (e.g., Zr, Cu, Zn)")
    q_parser.add_argument("--formula", help="Filter by chemical formula substring")
    q_parser.add_argument("--min-surface-area", type=float, help="Minimum surface area")
    q_parser.add_argument("--max-surface-area", type=float, help="Maximum surface area")
    q_parser.add_argument("--min-pore-size", type=float, help="Minimum pore size/diameter")
    q_parser.add_argument("--max-pore-size", type=float, help="Maximum pore size/diameter")
    q_parser.add_argument(
        "--output", default="data/query_results.csv",
        help="Output file path (default: data/query_results.csv)"
    )
    q_parser.add_argument(
        "--format", choices=["csv", "json", "both"], default="both",
        help="Output format (default: both)"
    )

    args = parser.parse_args()

    if args.command == "download":
        download(output=args.output, format=args.format)
    elif args.command == "query":
        query(
            metal=args.metal,
            min_surface_area=args.min_surface_area,
            max_surface_area=args.max_surface_area,
            min_pore_size=args.min_pore_size,
            max_pore_size=args.max_pore_size,
            formula=args.formula,
            output=args.output,
            format=args.format,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without errors**

Run: `cd /home/theoj/project/mofgen && python scripts/mp_data_extraction.py --help`
Expected: Shows help text with `download` and `query` subcommands.

- [ ] **Step 3: Commit**

```bash
git add scripts/mp_data_extraction.py
git commit -m "feat: add Materials Project data extraction script

Supports full download and filtered queries from MOFGen_2025 on MPContribs.
API key read from MP_API_KEY env var or .env file."
```

---

### Task 13: Test MP Data Extraction Script (Mocked API)

**Files:**
- Create: `tests/test_mp_data/test_mp_extraction.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_mp_data/test_mp_extraction.py`:

```python
"""Tests for MP data extraction — mocked MPContribs client."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from scripts.mp_data_extraction import get_api_key, download, query


class TestGetApiKey:
    def test_returns_key_from_env(self):
        with patch.dict("os.environ", {"MP_API_KEY": "test-key-123"}):
            assert get_api_key() == "test-key-123"

    def test_exits_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit):
                get_api_key()


@pytest.fixture
def mock_contributions():
    """Sample MPContribs response data."""
    return [
        {
            "identifier": "mp-12345",
            "formula": "Zn4O(BDC)3",
            "data.metal": "Zn",
            "data.surface_area": 1200.5,
            "data.pore_diameter": 8.3,
        },
        {
            "identifier": "mp-67890",
            "formula": "Zr6O4(BDC)6",
            "data.metal": "Zr",
            "data.surface_area": 2500.0,
            "data.pore_diameter": 12.1,
        },
        {
            "identifier": "mp-11111",
            "formula": "Cu3(BTC)2",
            "data.metal": "Cu",
            "data.surface_area": 800.0,
            "data.pore_diameter": 5.5,
        },
    ]


@pytest.fixture
def mock_client(mock_contributions):
    """Mock MPContribs client."""
    with patch("scripts.mp_data_extraction.get_client") as mock_get:
        client = MagicMock()
        client.get_contributions.return_value = mock_contributions
        mock_get.return_value = client
        yield client


class TestDownload:
    def test_saves_csv(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "out.csv")
            df = download(output=out, format="csv")
            assert Path(out).exists()
            assert len(df) == 3

    def test_saves_json(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "out.json")
            df = download(output=out, format="json")
            assert Path(out).exists()

    def test_saves_both(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "out.csv")
            df = download(output=out, format="both")
            assert (tmp_dir / "out.csv").exists()
            assert (tmp_dir / "out.json").exists()


class TestQuery:
    def test_filter_by_formula_metal(self, mock_client, tmp_dir):
        with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
            out = str(tmp_dir / "q.csv")
            df = query(formula="Zr", output=out, format="csv")
            # Should match the Zr entry
            assert len(df) >= 1
```

- [ ] **Step 2: Run tests**

Run: `cd /home/theoj/project/mofgen && pixi run test tests/test_mp_data/test_mp_extraction.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mp_data/test_mp_extraction.py
git commit -m "test: add MP data extraction tests with mocked API"
```

---

### Task 14: Add GitHub Actions CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow file**

```bash
mkdir -p /home/theoj/project/mofgen/.github/workflows
```

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  smoke-test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1

      - name: Install dependencies
        run: pixi install

      - name: Verify imports
        run: pixi run test-imports

      - name: Run tests
        run: pixi run test

  cuda-resolve:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install pixi
        uses: prefix-dev/setup-pixi@v0.8.1

      - name: Verify CUDA environment resolves
        run: pixi install -e cuda
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for cross-platform testing

Runs smoke tests on Linux, macOS, and Windows.
Verifies CUDA env resolves on Linux."
```

---

### Task 15: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README.md**

Replace the entire contents of `README.md` with:

```markdown
# MOFGen — AI-Driven Metal-Organic Framework Generation

Tools for Metal-Organic Framework (MOF) generation using diffusion models and AI agents.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

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
├── scripts/               # Utility scripts
│   └── mp_data_extraction.py  # Materials Project data download/query
├── tests/                 # Test suite
├── pixi.toml              # Environment & dependency config
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with CPU/CUDA instructions, MP data, troubleshooting"
```

---

### Task 16: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add entries for data output and env files**

Append to `.gitignore`:

```
# Data output
data/

# Environment files with secrets
.env

# Pixi lock (platform-specific, regenerated by pixi install)
# pixi.lock is kept for reproducibility — do not ignore

# pytest cache
.pytest_cache/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore for data output and env files"
```

---

### Task 17: Delete pixi.lock and Regenerate

**Files:**
- Delete and regenerate: `pixi.lock`

- [ ] **Step 1: Remove stale lock file and regenerate**

The existing `pixi.lock` was generated from the old `pixi.toml` and is now stale. It must be regenerated:

```bash
cd /home/theoj/project/mofgen
rm pixi.lock
pixi install
```

Expected: Clean install with no errors or warnings.

- [ ] **Step 2: Run import smoke test**

Run: `pixi run test-imports`
Expected: `All imports OK`

- [ ] **Step 3: Run full test suite**

Run: `pixi run test`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pixi.lock
git commit -m "chore: regenerate pixi.lock for new CPU/CUDA environment config"
```

---

### Task 18: Final Verification

- [ ] **Step 1: Run full test suite one more time**

```bash
cd /home/theoj/project/mofgen && pixi run test -v
```
Expected: All tests pass.

- [ ] **Step 2: Verify git status is clean**

```bash
git status
```
Expected: Clean working tree, no untracked files (except `data/` if generated).

- [ ] **Step 3: Review commit log**

```bash
git log --oneline -15
```
Expected: Clear sequence of commits matching each task above.
