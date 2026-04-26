"""LinkerGen MCP tool wrapper."""

from __future__ import annotations

from agents.agent_2_linkergen.linkergen_agent import LinkerGenAgent, LinkerGenConfig


def generate_linker(
    mode: str = "smiles",
    examples_file: str = "",
    num_linkers: int = 50,
    provider: str = "openai",
) -> str:
    """Invoke LinkerGen to generate new MOF linker candidates.

    Args:
        mode: ``"smiles"`` or ``"formula"``.
        examples_file: Path to file with example linkers.
        num_linkers: Number of linkers to generate.
        provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        Generated linkers as text.
    """
    if not examples_file.strip():
        raise ValueError("examples_file is required and must be a non-empty path.")
    config = LinkerGenConfig()
    agent = LinkerGenAgent(config=config, provider=provider)

    if mode == "formula":
        return agent.generate_formula_from_formula(
            examples_file=examples_file,
            num_linkers=num_linkers,
        )
    return agent.generate_smiles_from_smiles(
        examples_file=examples_file,
        num_linkers=num_linkers,
    )
