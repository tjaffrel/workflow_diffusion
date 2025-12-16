"""Linker Generation Agent for MOF Linker Discovery.

A simple LLM-based agent for generating new MOF linkers from example datasets.
Supports both SMILES-to-SMILES and Formula-to-Formula generation.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    raise ImportError(
        "langchain and langchain-openai are required. Install with: "
        "pip install langchain langchain-openai"
    )



@dataclass
class LinkerGenConfig:
    """Configuration for Linker Generation Agent."""
    model_name: str = "gpt-4"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    openai_api_key: Optional[str] = None


class LinkerGenAgent:
    """LLM agent for generating MOF linkers from example datasets."""
    
    def __init__(self, config: Optional[LinkerGenConfig] = None):
        """Initialize the linker generation agent.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or LinkerGenConfig()
        
        # Get API key from config or environment
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=api_key
        )
    
    def _load_examples(self, file_path: str) -> Optional[str]:
        """Load example linkers from a file.
        
        Args:
            file_path: Path to the example file (CSV, TXT, etc.)
            
        Returns:
            File contents as string, or None if file not found.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            return None
        return path.read_text(encoding="utf-8")
    
    def generate_smiles_from_smiles(
        self,
        examples_file: str,
        num_linkers: int = 100,
        output_file: Optional[str] = None
    ) -> str:
        """Generate new SMILES linkers from example SMILES.
        
        Args:
            examples_file: Path to file containing example SMILES
            num_linkers: Number of new linkers to generate
            output_file: Optional path to save results
            
        Returns:
            Generated SMILES as string
        """
        examples = self._load_examples(examples_file)
        if not examples:
            raise ValueError(f"Could not load examples from {examples_file}")
        
        system_prompt = (
            "You are a metal-organic framework (MOF) chemist. "
            "You want to come up with new chemical formulas for linkers of MOFs. "
            "The reply should only contain a list of new SMILES."
        )
        
        user_prompt = (
            f"Here are examples of linkers in SMILES: {examples}. "
            f"Now, generate {num_linkers} new different linkers of MOFs in SMILES, "
            "and put them into a list."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        result = response.content
        
        if output_file:
            Path(output_file).write_text(result, encoding="utf-8")
            print(f"Results saved to {output_file}")
        
        return result
    
    def generate_formula_from_formula(
        self,
        examples_file: str,
        num_linkers: int = 50,
        output_file: Optional[str] = None
    ) -> str:
        """Generate new chemical formula linkers from example formulas.
        
        Args:
            examples_file: Path to file containing example formulas
            num_linkers: Number of new linkers to generate
            output_file: Optional path to save results
            
        Returns:
            Generated formulas as string
        """
        examples = self._load_examples(examples_file)
        if not examples:
            raise ValueError(f"Could not load examples from {examples_file}")
        
        system_prompt = (
            "You are a metal-organic framework (MOF) chemist. "
            "You want to come up with new chemical formulas for linkers of MOFs. "
            "The reply should only contain a list of new chemical formulas."
        )
        
        user_prompt = (
            f"Here are examples of linkers in chemical formula: {examples}. "
            f"Now, generate {num_linkers} new different linkers of MOFs in "
            "chemical formula, and put them into a list."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        result = response.content
        
        if output_file:
            Path(output_file).write_text(result, encoding="utf-8")
            print(f"Results saved to {output_file}")
        
        return result
    
    def generate_custom(
        self,
        examples_file: str,
        system_prompt: str,
        user_prompt_template: str,
        num_linkers: int = 100,
        output_file: Optional[str] = None
    ) -> str:
        """Generate linkers with custom prompts.
        
        Args:
            examples_file: Path to file containing examples
            system_prompt: System prompt for the LLM
            user_prompt_template: User prompt template (use {examples} and {num_linkers})
            num_linkers: Number of linkers to generate
            output_file: Optional path to save results
            
        Returns:
            Generated linkers as string
        """
        examples = self._load_examples(examples_file)
        
        user_prompt = user_prompt_template.format(
            examples=examples,
            num_linkers=num_linkers
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        result = response.content
        
        if output_file:
            Path(output_file).write_text(result, encoding="utf-8")
        
        return result


def main():
    """Example usage of the LinkerGenAgent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MOF linkers using LLM")
    parser.add_argument(
        "--mode",
        choices=["smiles", "formula", "custom"],
        default="smiles",
        help="Generation mode"
    )
    parser.add_argument(
        "--examples",
        required=True,
        help="Path to examples file"
    )
    parser.add_argument(
        "--output",
        help="Output file path (optional)"
    )
    parser.add_argument(
        "--num-linkers",
        type=int,
        default=100,
        help="Number of linkers to generate"
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="LLM model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    config = LinkerGenConfig(
        model_name=args.model,
        temperature=args.temperature
    )
    agent = LinkerGenAgent(config)
    
    # Generate based on mode
    if args.mode == "smiles":
        result = agent.generate_smiles_from_smiles(
            args.examples,
            num_linkers=args.num_linkers,
            output_file=args.output
        )
    elif args.mode == "formula":
        result = agent.generate_formula_from_formula(
            args.examples,
            num_linkers=args.num_linkers,
            output_file=args.output
        )
    else:
        print("Custom mode requires additional prompt configuration.")
        return
    
    print("\nGenerated linkers:")
    print(result[:500] + "..." if len(result) > 500 else result)


if __name__ == "__main__":
    main()

