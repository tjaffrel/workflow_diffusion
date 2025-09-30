# MOF Generation Agents

This directory contains specialized LLM agents for Metal-Organic Framework (MOF) generation, analysis, and optimization.

## MOFMaster Agent

The primary MOF generation agent that provides three generation modes:

### 1. Basic Generation
Generate MOF structures without specific chemical constraints.

```python
from agents import MOFMaster

# Initialize with OpenAI API key
mof_master = MOFMaster()  # Uses OPENAI_API_KEY environment variable

# Generate basic structures
result = mof_master.generate_basic_structures(count=5)
print(f"Generated {result.success_count} structures in {result.generation_time:.2f} seconds")

for i, structure in enumerate(result.structures):
    print(f"Structure {i+1}: {structure.formula}")
    print(f"CIF content length: {len(structure.cif_content)} characters")
```

### 2. Metal-Specific Generation
Generate MOF structures with specific metal secondary building units (SBUs).

```python
# Generate Zn-based MOFs
zn_result = mof_master.generate_metal_specific_structures(metal="Zn", count=3)

# Generate Cu-based MOFs with constraints
cu_result = mof_master.generate_metal_specific_structures(
    metal="Cu", 
    count=2,
    pore_size="large",
    stability="high"
)
```

### 3. Composition-Specific Generation
Generate MOF structures with target chemical composition.

```python
# Define target composition
composition = {
    "Zn": 0.2,  # 20% Zn
    "C": 0.4,   # 40% C
    "H": 0.2,   # 20% H
    "O": 0.2    # 20% O
}

# Generate structures with target composition
comp_result = mof_master.generate_composition_specific_structures(
    composition=composition,
    count=2,
    application="CO2_capture"
)
```

## Setup

### 1. Environment Setup
```bash
# Install dependencies
pixi install

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Basic Usage
```python
from agents import MOFMaster

# Initialize agent
mof_master = MOFMaster()

# Generate structures
result = mof_master.generate_basic_structures(count=1)

# Access generated structures
for structure in result.structures:
    print(f"Formula: {structure.formula}")
    print(f"Metal SBU: {structure.metal_sbu}")
    print(f"Composition: {structure.composition}")
    print(f"CIF Content:\n{structure.cif_content}")
```

## Configuration

### OpenAI Model Selection
```python
# Use different OpenAI models
mof_master = MOFMaster(model="gpt-4o")        # Default
mof_master = MOFMaster(model="gpt-4")         # Alternative
mof_master = MOFMaster(model="gpt-3.5-turbo") # Faster, cheaper
```

### API Key Configuration
```python
# Method 1: Environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"
mof_master = MOFMaster()

# Method 2: Direct parameter
mof_master = MOFMaster(api_key="your-api-key-here")
```

## Output Structure

Each generated MOF structure contains:

- **`cif_content`**: CIF file content with atomic coordinates
- **`formula`**: Chemical formula
- **`metal_sbu`**: Primary metal in SBU (if applicable)
- **`composition`**: Elemental composition (if applicable)
- **`properties`**: Basic properties and descriptions
- **`generation_metadata`**: Generation parameters and metadata

## Error Handling

The agent includes robust error handling:

```python
try:
    result = mof_master.generate_basic_structures(count=5)
    if result.success_count > 0:
        print(f"Successfully generated {result.success_count} structures")
    else:
        print("No structures were generated")
except Exception as e:
    print(f"Error: {e}")
```

## Examples

### Complete Workflow Example
```python
from agents import MOFMaster

# Initialize
mof_master = MOFMaster()

# 1. Generate basic structures
print("Generating basic MOF structures...")
basic_result = mof_master.generate_basic_structures(count=3)
print(f"Generated {basic_result.success_count} basic structures")

# 2. Generate metal-specific structures
print("\nGenerating Zn-based MOF structures...")
zn_result = mof_master.generate_metal_specific_structures(metal="Zn", count=2)
print(f"Generated {zn_result.success_count} Zn-based structures")

# 3. Generate composition-specific structures
print("\nGenerating composition-specific structures...")
composition = {"Zn": 0.25, "C": 0.35, "H": 0.2, "O": 0.2}
comp_result = mof_master.generate_composition_specific_structures(
    composition=composition, 
    count=1
)
print(f"Generated {comp_result.success_count} composition-specific structures")

# 4. Analyze results
total_structures = (basic_result.success_count + 
                   zn_result.success_count + 
                   comp_result.success_count)
print(f"\nTotal structures generated: {total_structures}")
```

## Requirements

- Python 3.11+
- OpenAI API key
- Internet connection for API calls

## Dependencies

- `openai`: OpenAI API client
- `os`: Environment variable access
- `time`: Timing measurements
- `typing`: Type hints
- `dataclasses`: Data structures
- `enum`: Enumerations
