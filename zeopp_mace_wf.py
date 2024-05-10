from __future__ import annotations
from atomate2.forcefields.jobs import MACERelaxMaker
from dataclasses import dataclass, field
from jobflow import Flow, Maker, Job
from pymatgen.core import Structure
import os

from zeopp import run_zeopp_assessment

#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from pymatgen.core import Structure

@dataclass
class MofDiscovery(Maker):

    zeopp_path : str | None = None
    sorbates : list[str] | str = field(default_factory= lambda : ["N2", "CO2", "H2O"])
    zeopp_nproc : int = 1,
    ff_relax_maker : Maker = field(
        default_factory = lambda : MACERelaxMaker(
            model_kwargs = {
                #"model": "small",
                "default_dtype": "float32",
                "dispersion": True,
            }
        )
    )

    def make(
        self,
        structure : str | Structure,
    ) -> Flow:
        
        jobs : list[Job] = []

        cif_name = "zeopp_initial.cif"
        if isinstance(structure,str):
            cif_name = structure
            structure = Structure.from_file(structure)

        zeopp_init = run_zeopp_assessment(
            structure = structure,
            zeopp_path = self.zeopp_path,
            working_dir = None,
            sorbates = self.sorbates,
            cif_name = cif_name,
            nproc = self.zeopp_nproc
        )

        jobs += [zeopp_init]
        output = zeopp_init.output

        if zeopp_init.output["is_mof"]:
            # in case this job runs before the initial zeo++ job, uncomment the prev_dir kwarg below
            mace_relax = self.ff_relax_maker.make(structure)#, prev_dir = zeopp_init.output)
            jobs += [mace_relax]

            zeopp_final = run_zeopp_assessment(
                structure = mace_relax.output.structure,
                zeopp_path = self.zeopp_path,
                working_dir = None,
                sorbates = self.sorbates,
                cif_name = "zeopp_final.cif",
                nproc = self.zeopp_nproc
            )
            jobs += [zeopp_final]
            output = zeopp_final.output

        return Flow(jobs, output=output)
    
if __name__ == "__main__":

    from jobflow import run_locally

    mdj = MofDiscovery(
        zeopp_path = "/Users/aaronkaplan/Dropbox/postdoc_MP/software/zeo++-0.3/network",
        zeopp_nproc = 3
    ).make("IRMOF-1.cif")

    resp = run_locally(mdj)

    # for fireworks, uncomment below, comment out the `run_locally` line above:
    """
    from jobflow.managers.fireworks import flow_to_workflow
    from fireworks import LaunchPad

    fw = flow_to_workflow(mdj)

    lpad = LaunchPad.auto_load()
    # lpad = LaunchPad.from_file(<path to> my_launchpad.yaml)
    lpad.bulk_add_wfs([fw])
    """