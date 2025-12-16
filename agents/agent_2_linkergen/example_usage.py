"""Example usage of the Linker Generation Agent."""

from linker_gen_agent import LinkerGenAgent, LinkerGenConfig


def example_smiles_generation():
    """Example: Generate SMILES linkers from example SMILES."""
    print("Example 1: SMILES to SMILES Generation")
    print("-" * 50)
    
    # Initialize agent
    config = LinkerGenConfig(
        model_name="gpt-4",
        temperature=1.0
    )
    agent = LinkerGenAgent(config)
    
    # Generate new SMILES linkers
    result = agent.generate_smiles_from_smiles(
        examples_file="Example_Linker_SMILES.txt",
        num_linkers=100,
        output_file="Generated_Linker_SMILES.txt"
    )
    
    print(f"Generated {len(result)} characters of SMILES linkers")
    print(f"First 200 characters: {result[:200]}...")
    print()


def example_formula_generation():
    """Example: Generate formula linkers from example formulas."""
    print("Example 2: Formula to Formula Generation")
    print("-" * 50)
    
    # Initialize agent
    agent = LinkerGenAgent(LinkerGenConfig())
    
    # Generate new formula linkers
    result = agent.generate_formula_from_formula(
        examples_file="Example_Linker_Formula.txt",
        num_linkers=50,
        output_file="Generated_Linker_Formula.txt"
    )
    
    print(f"Generated {len(result)} characters of formula linkers")
    print(f"First 200 characters: {result[:200]}...")
    print()


def example_custom_generation():
    """Example: Custom generation with specialized prompts."""
    print("Example 3: Custom Generation")
    print("-" * 50)
    
    agent = LinkerGenAgent(LinkerGenConfig())
    
    # Custom generation for specific use case
    result = agent.generate_custom(
        examples_file="Example_Linker_SMILES.txt",
        system_prompt=(
            "You are an expert MOF chemist specializing in "
            "porous materials for gas storage applications."
        ),
        user_prompt_template=(
            "Given these example MOF linkers: {examples}\n\n"
            "Generate {num_linkers} new linker SMILES that are "
            "optimized for high surface area and CO2 adsorption capacity. "
            "Focus on linkers with aromatic rings and functional groups."
        ),
        num_linkers=100,
        output_file="custom_generated_linkers.txt"
    )
    
    print(f"Generated {len(result)} characters of custom linkers")
    print(f"First 200 characters: {result[:200]}...")
    print()


if __name__ == "__main__":
    # Run examples (comment out if you don't have API key set)
    try:
        example_smiles_generation()
        example_formula_generation()
        example_custom_generation()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Installed langchain and langchain-openai")
        print("3. Example files in the current directory")

