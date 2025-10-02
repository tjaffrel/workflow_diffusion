"""Example usage of MFOModeller agent."""

from pathlib import Path
from agent_4_qforge.mof_modeller import MFOModeller, MFOModellerConfig


def example_single_structure():
    """Example: Analyze a single MOF structure."""
    # Initialize MFOModeller agent
    config = MFOModellerConfig(
        zeopp_nproc=3,  # Use 3 processes for zeo++ analysis
        sorbates=["N2", "CO2", "H2O"],  # Analyze these sorbates
        run_local=True,  # Run locally (not remote)
        store_results=True  # Store results to files
    )
    
    modeller = MFOModeller(config)
    
    # Analyze a single MOF
    cif_path = "IRMOF-1.cif"  # Replace with actual CIF file
    
    print(f"Analyzing {cif_path}...")
    try:
        results = modeller.analyze_structure(cif_path)
        
        print("\n✓ Analysis completed!")
        print(f"Results stored for: {list(results.keys())}")
        
        # Print summary of results
        for job_name, result in results.items():
            print(f"\n{job_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if key != "structure":  # Don't print structure object
                        print(f"  {key}: {value}")
            else:
                print(f"  {result}")
                
        return results
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return None


def example_batch_analysis():
    """Example: Analyze multiple MOF structures in batch."""
    # Initialize MFOModeller config for batch processing
    config = MFOModellerConfig(
        zeopp_nproc=3,
        sorbates=["N2", "CO2", "H2O"],
        run_local=True,
        store_results=True
    )
    
    modeller = MFOModeller(config)
    
    # Analyze all CIF files in a directory
    cif_directory = "/home/theoj/project/diffusion/diffusion_MOF_v1/"  # Replace with actual directory
    
    print(f"Batch analyzing CIF files in: {cif_directory}")
    
    try:
        batch_results = modeller.batch_analyze(cif_directory)
        
        print(f"\n✓ Batch analysis completed!")
        print(f"Analyzed {len(batch_results)} structures")
        
        # Summary statistics
        successful = sum(1 for r in batch_results.values() if "error" not in r)
        failed = len(batch_results) - successful
        
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        # Show failed analyses
        if failed > 0:
            print("\nFailed analyses:")
            for mof_name, results in batch_results.items():
                if "error" in results:
                    print(f"  {mof_name}: {results['error']}")
        
        return batch_results
        
    except Exception as e:
        print(f"✗ Batch analysis failed: {e}")
        return None


def example_custom_sorbates():
    """Example: Analyze with custom sorbates and parameters."""
    # Custom configuration focusing on specific sorbates
    config = MFOModellerConfig(
        zeopp_path="/path/to/zeo++/executable",  # Custom zeo++ path if needed
        zeopp_nproc=2,  # Reduce parallelism for slower systems
        sorbates=["CO2"],  # Only analyze CO2 adsorption
        run_local=True,
        store_results=True
    )
    
    modeller = MFOModeller(config)
    
    # Analyze structure with custom parameters
    cif_path = "example.cif"
    
    print(f"Analyzing {cif_path} with CO2-only analysis...")
    
    try:
        results = modeller.analyze_structure(cif_path)
        print("✓ CO2 analysis completed!")
        return results
        
    except Exception as e:
        print(f"✗ CO2 analysis failed: {e}")
        return None


if __name__ == "__main__":
    """Run examples - uncomment the example you want to test."""
    
    # Example 1: Single structure analysis
    # example_single_structure()
    
    # Example 2: Batch analysis
    # example_batch_analysis()
    
    # Example 3: Custom sorbates
    # example_custom_sorbates()
    
    print("Examples ready to run. Uncomment one of the examples above.")
    print("Make sure to:")
    print("1. Have valid CIF files available")
    print("2. Install zeo++ and set ZEO_PATH environment variable")
    print("3. Install required dependencies: atomate2, jobflow, pymatgen")
