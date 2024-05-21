from __future__ import annotations
from atomate2.forcefields.jobs import MACERelaxMaker
from dataclasses import dataclass, field
from jobflow import Flow, Maker, job, Response
from pymatgen.core import Structure

from zeopp import run_zeopp_assessment

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pymatgen.core import Structure

@dataclass
class MofDiscovery(Maker):

    name : str = "MOF discovery zeo++ / MACE"
    zeopp_path : str | None = None
    sorbates : list[str] | str = field(default_factory = lambda : ["N2", "CO2", "H2O"])
    zeopp_nproc : int = 1
    ff_relax_maker : Maker = field(
        default_factory = lambda : MACERelaxMaker(
            calculator_kwargs = {
                #"model": "small",
                "default_dtype": "float64",
                "dispersion": True,
            },
            task_document_kwargs = {
                "store_trajectory": "no",
                "ionic_step_data": ("energy",)
            }
        )
    )

    @job
    def make(
        self,
        structure : Structure,
        mof_assessment : dict | None = None,
    ) -> Response:
                
        replace = None
        output = mof_assessment
        if mof_assessment is None:

            zeopp_init = run_zeopp_assessment(
                structure = structure,
                zeopp_path = self.zeopp_path,
                working_dir = None,
                sorbates = self.sorbates,
                cif_name = "zeopp_initial.cif",
                nproc = self.zeopp_nproc
            )
            zeopp_init.name = "zeo++ input structure"
            #zeopp_init.metadata = {"job_type": "zeo++"}

            mace_jobs = self.make(
                structure = structure,
                mof_assessment = {
                    "zeo++ initial": zeopp_init.output,
                }
            )

            output = mace_jobs.output
            replace = Flow([zeopp_init, mace_jobs], output = output)
        
        elif (
            mof_assessment.get("zeo++ initial",{}).get("is_mof", False)
            and ("MACE_force_converged" not in mof_assessment)
        ):

            mace_relax = self.ff_relax_maker.make(structure)
            mace_relax.name = "MACE relax"
            mof_assessment.update({
                "MACE_force_converged": mace_relax.output.is_force_converged
            })
            output = mace_relax.output
            recursive = self.make(
                structure = mace_relax.output.structure,
                mof_assessment = mof_assessment,
            )
            replace = Flow([mace_relax, recursive], output=output)

        elif mof_assessment.get("MACE_force_converged",False):
            zeopp_final = run_zeopp_assessment(
                structure = structure,
                zeopp_path = self.zeopp_path,
                working_dir = None,
                sorbates = self.sorbates,
                cif_name = "zeopp_final.cif",
                nproc = self.zeopp_nproc
            )
            zeopp_final.name = "zeo++ mace-relaxed structure"

            output = zeopp_final.output
            replace = zeopp_final
        
        return Response(replace=replace, output = output)