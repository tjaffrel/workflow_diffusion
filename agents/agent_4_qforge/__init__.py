"""
QForge Agent - MOF Analysis and Optimization

This package provides clean Python classes for MOF analysis workflows,
replacing the original Jupyter notebook implementation.

Main components:
- MFOModeller: Main agent class for MOF analysis
- MofDiscovery: Workflow for zeo++ analysis and MACE relaxation  
- ZeoPlusPlus: Zeo++ analysis functionality
"""

from .mof_modeller import MFOModeller, MFOModellerConfig
from .mof_discovery import MofDiscovery
from .zeopp_analyzer import ZeoPlusPlus, run_zeopp_assessment

__all__ = [
    "MFOModeller",
    "MFOModellerConfig", 
    "MofDiscovery",
    "ZeoPlusPlus",
    "run_zeopp_assessment"
]
