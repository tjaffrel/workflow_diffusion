#!/usr/bin/env python3
"""Example script demonstrating MOFMaster agent usage.

Shows how to use MOFMaster in three generation modes (basic, metal-specific,
composition-specific) with either OpenAI or Anthropic as the LLM provider.

The pipeline originally used gpt-4o; default is now gpt-4.1 for OpenAI and
claude-sonnet-4-20250514 for Anthropic.

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="your_key"
    pixi run python example_mof_generation.py

    # Anthropic (Claude) — works without an OpenAI key
    export ANTHROPIC_API_KEY="your_key"
    pixi run python example_mof_generation.py --provider anthropic
"""

import argparse
import os

from agents import MOFMaster

ENV_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def main():
    parser = argparse.ArgumentParser(description="MOFMaster example")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    args = parser.parse_args()

    env_var = ENV_VARS[args.provider]
    if not os.getenv(env_var):
        print(f"Error: {env_var} environment variable not set.")
        print(f"export {env_var}='your-api-key-here'")
        return

    print(f"Initializing MOFMaster with provider={args.provider!r}...")
    mof_master = MOFMaster(provider=args.provider)
    print("MOFMaster initialized successfully\n")

    # 1. Generate basic structures
    print("=" * 50)
    print("1. GENERATING BASIC MOF STRUCTURES")
    print("=" * 50)

    try:
        basic_response = mof_master.generate_basic_structures(count=2)
        print(f"Generated {basic_response.success_count} basic structures in {basic_response.generation_time:.2f}s")

        for i, structure in enumerate(basic_response.structures):
            print(f"\nStructure {i+1}:")
            print(f"  Formula: {structure.formula}")
            print(f"  CIF length: {len(structure.cif_content)} characters")
            print(f"  Properties: {structure.properties}")

    except Exception as e:
        print(f"Error generating basic structures: {e}")

    # 2. Generate metal-specific structures
    print("\n" + "=" * 50)
    print("2. GENERATING METAL-SPECIFIC MOF STRUCTURES")
    print("=" * 50)

    try:
        metal_response = mof_master.generate_metal_specific_structures(metal="Zn", count=2)
        print(f"Generated {metal_response.success_count} Zn-based structures in {metal_response.generation_time:.2f}s")

        for i, structure in enumerate(metal_response.structures):
            print(f"\nStructure {i+1}:")
            print(f"  Formula: {structure.formula}")
            print(f"  Metal SBU: {structure.metal_sbu}")
            print(f"  CIF length: {len(structure.cif_content)} characters")
            print(f"  Properties: {structure.properties}")

    except Exception as e:
        print(f"Error generating metal-specific structures: {e}")

    # 3. Generate composition-specific structures
    print("\n" + "=" * 50)
    print("3. GENERATING COMPOSITION-SPECIFIC MOF STRUCTURES")
    print("=" * 50)

    try:
        composition = {"Zn": 0.2, "C": 0.4, "H": 0.2, "O": 0.2}
        comp_response = mof_master.generate_composition_specific_structures(
            composition=composition,
            count=1,
        )
        print(f"Generated {comp_response.success_count} composition-specific structures in {comp_response.generation_time:.2f}s")

        for i, structure in enumerate(comp_response.structures):
            print(f"\nStructure {i+1}:")
            print(f"  Formula: {structure.formula}")
            print(f"  Target composition: {structure.composition}")
            print(f"  CIF length: {len(structure.cif_content)} characters")
            print(f"  Properties: {structure.properties}")

    except Exception as e:
        print(f"Error generating composition-specific structures: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_structures = 0
    total_time = 0.0

    try:
        total_structures += basic_response.success_count
        total_time += basic_response.generation_time
    except NameError:
        pass

    try:
        total_structures += metal_response.success_count
        total_time += metal_response.generation_time
    except NameError:
        pass

    try:
        total_structures += comp_response.success_count
        total_time += comp_response.generation_time
    except NameError:
        pass

    print(f"Total structures generated: {total_structures}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Average time per structure: {total_time/max(total_structures, 1):.2f}s")

    print("\nMOF generation example completed successfully!")


if __name__ == "__main__":
    main()
