# Agent 4 QForge - MOF Analysis & Optimization

Clean Python replacement for `run_jobs.ipynb` workflow. Combines zeo++ pore analysis with MACE force field relaxation for MOF characterization.

## Workflow

```
Input CIF → ZeO++ Analysis → MOF Validation → MACE Relaxation → Final ZeO++ → Results
```

1. **ZeO++ Initial**: Analyze pore accessibility (PLD, LCD, volume fractions)
2. **MOF Validation**: Check structure meets MOF criteria
3. **MACE Relaxation**: Optimize atomic positions using force field
4. **ZeO++ Final**: Re-analyze pores after relaxation
5. **Results**: Return optimized structure + analysis data

## Requirements

### Required Software
- **zeo++**: Pore analysis software
  - Download: https://github.com/lsmo/zeo++
  - Set `ZEO_PATH` environment variable to executable path
- **atomate2**: MOF force field calculations
- **MACE models**: Downloaded automatically or configure paths

### Configuration Files

#### jobflow_config.yaml (Optional)
```yaml
JB_NODES:
  - name: local
    worker: local

# For remote execution
# BLAZES:
#   - name: fireworks
#     manager: fireworks
#     params:
#       host: localhost
#       name: qforge_jobs
```

#### Environment Setup
```bash
export ZEO_PATH="/path/to/zeo++/network"
export JOBFLOW_CONFIG_FILE="jobflow_config.yaml"
```

## Usage

### Basic Usage
```python
from agent_4_qforge import MFOModeller, MFOModellerConfig

# Configure agent
config = MFOModellerConfig(
    zeopp_path=None,           # Uses ZEO_PATH env var
    zeopp_nproc=3,            # Parallel zeo++ processes
    sorbates=["N2", "CO2"],   # Analyzed sorbates
    store_results=True        # Save output files
)

# Initialize agent
modeller = MFOModeller(config)

# Single analysis
results = modeller.analyze_structure("IRMOF-1.cif")
print(f"Is MOF: {results['zeo++ initial']['is_mof']}")
```

### Batch Processing
```python
# Analyze directory of CIF files
batch_results = modeller.batch_analyze("/path/to/cif/directory/")

for name, result in batch_results.items():
    if "error" not in result:
        print(f"{name}: PLD={result['zeo++ final']['N2']['PLD']:.2f} Å")
```

### Advanced: Direct Workflow Access
```python
from agent_4_qforge import MofDiscovery
from pymatgen.core import Structure

# Direct workflow usage
structure = Structure.from_file("MOF.cif")
workflow = MofDiscovery(zeopp_nproc=4).make(structure)

# Run with local manager
from jobflow import run_locally
response = run_locally(workflow)
results = response.get(workflow.uuid)[1]
```

## Output Data Structure

```python
{
    "zeo++ initial": {
        "is_mof": bool,
        "N2": {"PLD": float, "LCD": float, "POAV_A^3": float, ...},
        "CO2": {"PLD": float, "LCD": float, ...},
        "H2O": {"PLD": float, "LCD": float, ...}
    },
    "MACE_force_converged": bool,
    "zeo++ mace-relaxed structure": {
        "N2": {"PLD": float, ...},  # Post-relaxation values
        "structure": pymatgen.Structure  # Optimized structure
    }
}
```

## MOF Criteria

Structure qualifies as MOF if all criteria met:
- PLD > 2.5 Å
- POAV volume fraction > 0.3
- POAV > PONAV (volume comparisons)

## Troubleshooting

### Common Issues
- **zeo++ not found**: Set `ZEO_PATH` environment variable
- **MACE model errors**: Ensure atomate2[forcefields] installed
- **Memory issues**: Reduce `zeopp_nproc` for large structures
- **Analysis failure**: Check CIF file validity and atom types

### Logs & Output
- Results saved to `mof_analysis_<name>/` directories
- JobFlow logs contain detailed execution traces
- zeo++ raw output stored in working directories