# Linker Generation Agent

A standalone LLM agent for generating new Metal-Organic Framework (MOF) linkers from example datasets. Supports SMILES-to-SMILES and Formula-to-Formula generation using LangChain and OpenAI.

## Installation

Install dependencies:
```bash
pixi install
# or
pip install langchain langchain-openai openai
```

Set OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Python API

```python
from agents.agent_2_linkergen.linker_gen_agent import LinkerGenAgent, LinkerGenConfig

# Initialize agent
agent = LinkerGenAgent(LinkerGenConfig(model_name="gpt-4", temperature=1.0))

# Generate SMILES linkers
result = agent.generate_smiles_from_smiles(
    examples_file="Example_Linker_SMILES.txt",
    num_linkers=100,
    output_file="Generated_Linker_SMILES.txt"
)

# Generate formula linkers
result = agent.generate_formula_from_formula(
    examples_file="Example_Linker_Formula.txt",
    num_linkers=50,
    output_file="Generated_Linker_Formula.txt"
)
```

### Command Line

```bash
# Generate SMILES linkers
python -m agents.agent_2_linkergen.linker_gen_agent \
    --mode smiles \
    --examples Example_Linker_SMILES.txt \
    --num-linkers 100 \
    --output Generated_Linker_SMILES.txt

# Generate formula linkers
python -m agents.agent_2_linkergen.linker_gen_agent \
    --mode formula \
    --examples Example_Linker_Formula.txt \
    --num-linkers 50 \
    --output Generated_Linker_Formula.txt
```

## Configuration

- `model_name`: LLM model provided by OpenAI
- `temperature`: Sampling temperature (default: 1.0, higher = more creative)
- `openai_api_key`: API key (can use `OPENAI_API_KEY` env var)

## Example Files

The agent expects example linkers:
- `Example_Linker_SMILES.txt`: SMILES representations
- `Example_Linker_Formula.txt`: Chemical formulas

Files can contain one linker per line, comma-separated values, or any text format that provides context to the LLM.
