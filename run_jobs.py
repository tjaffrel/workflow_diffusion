
from __future__ import annotations
from pymatgen.core import Structure

from zeopp_mace_wf import MofDiscovery

def get_uuid_from_job(job, dct):
    if hasattr(job,"jobs"):
        for j in job.jobs:
            get_uuid_from_job(j,dct)
    else:
        dct[job.uuid] = job.name

def get_uuid_from_response(job, response) -> dict:
    uuid_to_name = {}
    get_uuid_from_job(job, uuid_to_name)
    for resp in response.values():
        if hasattr(resp[1],"replace") and resp[1].replace:
            for job in resp[1].replace:
                uuid_to_name[job.uuid] = job.name
    return uuid_to_name

def _adk_debug_locally(
    cif_name : str = "IRMOF-1.cif"
) -> dict:
    from jobflow import run_locally, Response

    structure = Structure.from_file(cif_name)
    mdj = MofDiscovery(zeopp_nproc = 3).make(structure = structure)

    response = run_locally(mdj)
    uuid_to_name = get_uuid_from_response(mdj, response)

    return {
        name: response.get(uuid)[1] 
        for uuid, name in uuid_to_name.items() 
        if not isinstance(response[uuid],Response)
    } 
   
def _adk_debug_remotely(
    cif_name : str = "IRMOF-1.cif",
    lpad_file : str = "/Users/aaronkaplan/fw_config/wf_dev/my_launchpad.yaml"
) -> None:
    from fireworks import LaunchPad
    from jobflow.managers.fireworks import flow_to_workflow

    structure = Structure.from_file(cif_name)
    mdj = MofDiscovery(zeopp_nproc = 3).make(structure = structure)

    mof_name = cif_name.split(".cif")[0]
    job_meta = {"MOF": mof_name, "job_info": "mof discovery"}

    mdj.update_metadata( job_meta)
    mdj.append_name( mof_name + " ",prepend=True)

    fw = flow_to_workflow(mdj)
    fw.metadata = job_meta
    
    lpad = LaunchPad.from_file(lpad_file)
    lpad.add_wf(fw)

def tji_run_all_cifs(
    cif_path : str = "/home/theoj/project/diffusion/diffusion_MOF_v1/",
    lpad_path : str = "/home/theoj/fw_config/my_launchpad.yaml", #"/global/homes/t/theoj/fw_configs/zeopp/my_launchpad.yaml"
) -> None:
    from fireworks import LaunchPad
    from glob import glob
    from jobflow.managers.fireworks import flow_to_workflow

    list_cif = glob(f"{cif_path}/*.cif")
    wfs = []
    for cif in list_cif[:1]:
        mof_name = cif.split("/")[-1].split(".")[0]
        job_meta = {"MOF": mof_name, "job_info": "mof discovery"}
        structure = Structure.from_file(cif)
        mdj = MofDiscovery(
            zeopp_nproc = 3
        ).make(
            structure = structure,
        )
        mdj.update_metadata( job_meta)
        mdj.append_name( mof_name + " ",prepend=True)

        fw = flow_to_workflow(mdj)
        fw.metadata = job_meta
        wfs.append(fw)

    
    lpad = LaunchPad.from_file(lpad_path)
    lpad.bulk_add_wfs(wfs)

if __name__ == "__main__":

    #flow_output = _adk_debug_locally()
    _adk_debug_remotely()