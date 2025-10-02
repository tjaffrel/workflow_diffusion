"""MOF Modeller Agent - Clean Python class replacing run_jobs.ipynb workflow."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from pymatgen.core import Structure
from jobflow import run_locally, Response
from atomate2.utils.testing import get_job_uuid_name_map

from .mof_discovery import MofDiscovery


@dataclass
class MFOModellerConfig:
    """Configuration for MOF Modeller Agent."""
    zeopp_path: str | None = None
    zeopp_nproc: int = 3
    sorbates: list[str] | str = ["N2", "CO2", "H2O"]
    run_local: bool = True
    store_results: bool = True


class MFOModeller:
    """
    MOF Modeller Agent for zeo++ analysis and MACE relaxation.
    
    Replaces run_jobs.ipynb functionality. Takes CIF file, performs zeo++ 
    assessment + MACE relaxation, returns optimized structure + properties.
    """
    
    def __init__(self, config: MFOModellerConfig | None = None):
        self.config = config or MFOModellerConfig()
        
    def analyze_structure(
        self, 
        cif_path: str | Path, 
        structure: Structure | None = None
    ) -> dict[str, Any]:
        """Analyze MOF structure using zeo++ and MACE workflow."""
        cif_path = Path(cif_path)
        
        # Load structure if not provided
        if structure is None:
            try:
                structure = Structure.from_file(str(cif_path))
            except Exception as e:
                raise ValueError(f"Could not parse CIF {cif_path}: {e}")
        
        # Create MOF discovery workflow
        mdj = MofDiscovery(
            zeopp_path=self.config.zeopp_path,
            zeopp_nproc=self.config.zeopp_nproc,
            sorbates=self.config.sorbates
        ).make(structure=structure)
        
        if self.config.run_local:
            return self._run_locally(mdj, cif_path.stem)
        else:
            raise NotImplementedError("Remote execution requires LaunchPad setup")
    
    def _run_locally(self, mdj: Any, mof_name: str) -> dict[str, Any]:
        """Run MOF discovery workflow locally."""
        # Update metadata with MOF name
        mdj.update_metadata({"MOF": mof_name, "job_info": "mof discovery"})
        mdj.append_name(mof_name + " ", prepend=True)
        
        # Run workflow locally
        response = run_locally(mdj)
        uuid_to_name = get_job_uuid_name_map(mdj, response)
        
        # Map results by job name, excluding Response objects
        results = {
            name: response.get(uuid)[1]
            for uuid, name in uuid_to_name.items()
            if not isinstance(response[uuid], Response)
        }
        
        if self.config.store_results:
            self._store_results(results, mof_name)
            
        return results
    
    def _store_results(self, results: dict[str, Any], mof_name: str) -> None:
        """Store analysis results for the MOF."""
        output_dir = Path(f"mof_analysis_{mof_name}")
        output_dir.mkdir(exist_ok=True)
        
        # Save summary report
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"MOF Analysis Results for: {mof_name}\n")
            f.write("=" * 50 + "\n")
            
            for job_name, result in results.items():
                f.write(f"\n{job_name}:\n")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key != "structure":  # Skip structure objects
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {result}\n")
        
        print(f"Analysis results stored in: {output_dir}")
    
    def batch_analyze(self, cif_directory: str | Path) -> dict[str, dict[str, Any]]:
        """Analyze multiple CIF files in batch."""
        cif_dir = Path(cif_directory)
        cif_files = list(cif_dir.glob("*.cif"))
        
        if not cif_files:
            raise ValueError(f"No CIF files found in {cif_directory}")
        
        results = {}
        
        for cif_file in cif_files:
            mof_name = cif_file.stem
            print(f"Analyzing {mof_name}...")
            
            try:
                analysis_results = self.analyze_structure(cif_file)
                results[mof_name] = analysis_results
                print(f"✓ Completed analysis for {mof_name}")
                
            except Exception as e:
                print(f"✗ Failed analysis for {mof_name}: {e}")
                results[mof_name] = {"error": str(e)}
        
        return results