"""MOF Generation Agent Framework

This module provides a framework for creating specialized LLM agents for MOF generation,
analysis, and optimization tasks.

Example:
    from agents import MOFMaster
    
    # Create MOF generation agent
    mof_master = MOFMaster()
    
    # Generate basic structures
    result = mof_master.generate_basic_structures(count=5)
    
    # Generate metal-specific structures
    zn_mofs = mof_master.generate_metal_specific_structures(metal="Zn", count=3)
    
    # Generate composition-specific structures
    composition = {"Zn": 0.2, "C": 0.4, "H": 0.2, "O": 0.2}
    comp_mofs = mof_master.generate_composition_specific_structures(composition=composition, count=2)
"""

from .mof_master import MOFMaster

__all__ = [
    "MOFMaster",
]
