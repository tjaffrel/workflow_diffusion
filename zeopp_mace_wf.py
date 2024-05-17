from __future__ import annotations
from atomate2.forcefields.jobs import MACERelaxMaker
from dataclasses import dataclass, field
from jobflow import Flow, Maker, job, Response
from pymatgen.core import Structure
import os

from zeopp import run_zeopp_assessment

#from typing import TYPE_CHECKING
#if TYPE_CHECKING:
#    from pymatgen.core import Structure

@dataclass
class MofDiscovery(Maker):

    name : str = "MOF discovery zeo++ / MACE"
    zeopp_path : str | None = None
    sorbates : list[str] | str = field(default_factory= lambda : ["N2", "CO2", "H2O"])
    zeopp_nproc : int = 1,
    ff_relax_maker : Maker = field(
        default_factory = lambda : MACERelaxMaker(
            calculator_kwargs = {
                #"model": "small",
                "default_dtype": "float32",
                "dispersion": True,
            }
        )
    )

    @job
    def make(
        self,
        structure : str | Structure,
        mof_assessment : dict | None = None,
        cif_name : str | None = "zeopp_initial.cif",
        metadata : dict | None = None,
        aux_name : str | None = None,
    ) -> Flow:
        
                
        if mof_assessment is None:
            if isinstance(structure, str):
                cif_name = os.path.basename(structure)
                structure = Structure.from_file(structure)

            zeopp_init = run_zeopp_assessment(
                structure = structure,
                zeopp_path = self.zeopp_path,
                working_dir = None,
                sorbates = self.sorbates,
                cif_name = cif_name,
                nproc = self.zeopp_nproc
            )
            zeopp_init.name = "zeo++ input structure"
            zeopp_init.metadata = {"job_type": "zeo++"}

            mace_jobs = self.make(
                structure = structure,
                mof_assessment = zeopp_init.output,
                cif_name = cif_name,
                metadata = metadata,
                aux_name = aux_name,
            )

            new_flow = Flow([zeopp_init, mace_jobs], output = mace_jobs.output)
            
            if metadata is not None:
                new_flow.update_metadata(metadata)

            if aux_name is not None:
                new_flow.append_name(aux_name,prepend=True)

            return Response(replace=new_flow, output = mace_jobs.output)

        elif mof_assessment.get("is_mof", False):

            mace_relax = self.ff_relax_maker.make(structure)
            mace_relax.metadata = {"job_type": "mace-relax"}

            zeopp_final = run_zeopp_assessment(
                structure = mace_relax.output.structure,
                zeopp_path = self.zeopp_path,
                working_dir = None,
                sorbates = self.sorbates,
                cif_name = cif_name,
                nproc = self.zeopp_nproc
            )
            zeopp_final.name = "zeo++ MACE-relaxed structure"
            zeopp_final.metadata = {"job_type": "zeo++"}

            new_flow = Flow([mace_relax, zeopp_final],output=zeopp_final.output)
            return Response(replace = new_flow, output=zeopp_final.output)
        
        return mof_assessment

def get_uuid_from_job(job, dct):
    if hasattr(job,"jobs"):
        for j in job.jobs:
            get_uuid_from_job(j,dct)
    else:
        dct[job.uuid] = job.name

if __name__ == "__main__":

    from jobflow import run_locally
    from jobflow.managers.fireworks import flow_to_workflow
    from glob import glob

    list_cif = glob("/home/theoj/project/diffusion/diffusion_MOF_v1/*.cif")
    wfs = []
    for cif in list_cif[:10]:
        mof_name = cif.split("/")[-1].split(".")[0]
        job_meta = {"MOF": mof_name, "job_info": "mof discovery"}
        mdj = MofDiscovery(
            zeopp_nproc = 3
        ).make(
            structure = cif,
            metadata = job_meta,
            aux_name = mof_name + " "
        )
        mdj.update_metadata( job_meta)
        mdj.append_name( mof_name + " ",prepend=True)
        fw = flow_to_workflow(mdj)
        fw.metadata = job_meta
        wfs.append(fw)

    from fireworks import LaunchPad

    lpad = LaunchPad.from_file("/home/theoj/fw_config/my_launchpad.yaml")#"/global/homes/t/theoj/fw_configs/zeopp/my_launchpad.yaml")
    lpad.bulk_add_wfs(wfs)

    """

    response = run_locally(mdj)

    uuid_to_name = {}
    get_uuid_from_job(mdj,uuid_to_name)
    for resp in response.values():
        if hasattr(resp[1],"replace") and resp[1].replace:
            for job in resp[1].replace:
                uuid_to_name[job.uuid] = job.name

    flow_output = {
        name: response.get(uuid)[1] 
        for uuid, name in uuid_to_name.items() 
        if not isinstance(response[uuid],Response)
    }
    """
    
    # for fireworks, uncomment below, comment out the `run_locally` line above:
    #from jobflow.managers.fireworks import flow_to_workflow
    #from fireworks import LaunchPad

    #fw = flow_to_workflow(mdj)

    #lpad = LaunchPad.auto_load()
    #lpad = LaunchPad.from_file("/home/theoj/fw_config/my_launchpad.yaml")
    #lpad.bulk_add_wfs([fw])
