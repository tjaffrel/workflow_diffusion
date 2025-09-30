#!/usr/bin/env python3
"""
Example script demonstrating MOFMaster agent usage.

This script shows how to use the MOFMaster agent to generate MOF structures
in three different modes: basic, metal-specific, and composition-specific.

Requirements:
- Set OPENAI_API_KEY environment variable
- Run: pixi run python example_mof_generation.py
"""

import os
from agents import MOFMaster


def main():
    """Main function demonstrating MOFMaster usage."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize MOFMaster
    print("Initializing MOFMaster agent...")
    mof_master = MOFMaster()
    print("✓ MOFMaster initialized successfully\n")
    
    # 1. Generate basic structures
    print("=" * 50)
    print("1. GENERATING BASIC MOF STRUCTURES")
    print("=" * 50)
    
    try:
        basic_response = mof_master.generate_basic_structures(count=2)
        print(f"✓ Generated {basic_response.success_count} basic structures in {basic_response.generation_time:.2f} seconds")
        
        for i, structure in enumerate(basic_response.structures):
            print(f"\nStructure {i+1}:")
            print(f"  Formula: {structure.formula}")
            print(f"  CIF length: {len(structure.cif_content)} characters")
            print(f"  Properties: {structure.properties}")
            
    except Exception as e:
        print(f"✗ Error generating basic structures: {e}")
    
    # 2. Generate metal-specific structures
    print("\n" + "=" * 50)
    print("2. GENERATING METAL-SPECIFIC MOF STRUCTURES")
    print("=" * 50)
    
    try:
        metal_response = mof_master.generate_metal_specific_structures(metal="Zn", count=2)
        print(f"✓ Generated {metal_response.success_count} Zn-based structures in {metal_response.generation_time:.2f} seconds")
        
        for i, structure in enumerate(metal_response.structures):
            print(f"\nStructure {i+1}:")
            print(f"  Formula: {structure.formula}")
            print(f"  Metal SBU: {structure.metal_sbu}")
            print(f"  CIF length: {len(structure.cif_content)} characters")
            print(f"  Properties: {structure.properties}")
            
    except Exception as e:
        print(f"✗ Error generating metal-specific structures: {e}")
    
    # 3. Generate composition-specific structures
    print("\n" + "=" * 50)
    print("3. GENERATING COMPOSITION-SPECIFIC MOF STRUCTURES")
    print("=" * 50)
    
    try:
        composition = {"Zn": 0.2, "C": 0.4, "H": 0.2, "O": 0.2}
        comp_response = mof_master.generate_composition_specific_structures(
            composition=composition, 
            count=1
        )
        print(f"✓ Generated {comp_response.success_count} composition-specific structures in {comp_response.generation_time:.2f} seconds")
        
        for i, structure in enumerate(comp_response.structures):
            print(f"\nStructure {i+1}:")
            print(f"  Formula: {structure.formula}")
            print(f"  Target composition: {structure.composition}")
            print(f"  CIF length: {len(structure.cif_content)} characters")
            print(f"  Properties: {structure.properties}")
            
    except Exception as e:
        print(f"✗ Error generating composition-specific structures: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    total_structures = 0
    total_time = 0.0
    
    try:
        total_structures += basic_response.success_count
        total_time += basic_response.generation_time
    except:
        pass
    
    try:
        total_structures += metal_response.success_count
        total_time += metal_response.generation_time
    except:
        pass
    
    try:
        total_structures += comp_response.success_count
        total_time += comp_response.generation_time
    except:
        pass
    
    print(f"Total structures generated: {total_structures}")
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Average time per structure: {total_time/max(total_structures, 1):.2f} seconds")
    
    print("\n✓ MOF generation example completed successfully!")


if __name__ == "__main__":
    main()
