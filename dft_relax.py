from pymatgen.io.vasp.sets import MP24RelaxSet, MP24StaticSet
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPMetaGGAStaticMaker,
    MPPreRelaxMaker,
)
from atomate2.vasp.flows.mp import (
    MPMetaGGADoubleRelaxMaker,
    MPMetaGGADoubleRelaxStaticMaker,
    MP24DoubleRelaxStaticMaker,
)
from custodian.vasp.utils import _estimate_num_k_points_from_kspacing
from math import ceil
from monty.serialization import loadfn
import numpy as np
import os
from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints
import warnings
from zipfile import ZipFile
from jobflow import Maker, Flow

from jobflow.managers.fireworks import flow_to_workflow

# catch CIF warnings like exceptions
warnings.filterwarnings("error", message="Incorrect stoichiometry")

global_kspacing = 0.32

incar_updates_static = {
    "NCORE": 1,
    "NSIM": 8,
    "KPAR": 2,
    "ALGO": "Normal",
    "IVDW": 13,  # DFTD4
    "LREAL": False,
    "LMAXMIX": 6,
    "LELF": False,
    "LVTOT": False,
}

incar_updates_relax = {
    **incar_updates_static,
    "EDIFFG": -0.05,
    "IBRION": 2,
    "ISIF": 3,
    "NSW": 99,
    "LWAVE": True,
}


def mof_flow(
    structure,
    pre_relax_maker: Maker | None = ForceFieldRelaxMaker(
        force_field_name="MACE",
        calculator_kwargs={
            "model": "/global/cfs/cdirs/matgen/esoteric/share/roberta/mace_omat/mace_models/mace-omat-0-medium.model",
            "dispersion": True,
        },
    ),
):
    nkpts = np.prod(_estimate_num_k_points_from_kspacing(structure, global_kspacing))
    user_kpoints_settings = Kpoints().as_dict()
    user_incar_settings = {}
    if nkpts < 4:
        # need 4 for tetrahedron
        user_incar_settings = {"ISMEAR": 0, "SIGMA": 0.05}

    """
    if nkpts == 1:
        # ensure we use gamma-point only vasp
        user_kpoints_settings = Kpoints().as_dict()
    else:
        user_incar_settings = {"KSPACING": global_kspacing}
    """

    pbe_relax_set = MP24RelaxSet(
        user_incar_settings={
            **incar_updates_relax,
            **user_incar_settings,
            "EDIFFG": -0.1,
        },
        user_kpoints_settings=user_kpoints_settings,
        xc_functional="PBE",
        dispersion="D4",
    )

    r2scan_relax_set = MP24RelaxSet(
        user_incar_settings={
            **incar_updates_relax,
            **user_incar_settings,
        },
        user_kpoints_settings=user_kpoints_settings,
        xc_functional="r2SCAN",
        dispersion="D4",
    )

    r2scan_static_set = MP24StaticSet(
        user_incar_settings={
            **incar_updates_static,
            **user_incar_settings,
        },
        user_kpoints_settings=user_kpoints_settings,
        xc_functional="r2SCAN",
        dispersion="D4",
    )

    double_relax = MPMetaGGADoubleRelaxMaker(
        relax_maker1=MPPreRelaxMaker(
            input_set_generator=pbe_relax_set, name="PBE-D4 relax"
        ),
        relax_maker2=MPMetaGGARelaxMaker(
            input_set_generator=r2scan_relax_set, name="r2SCAN-D4 relax"
        ),
        name="PBE-D4 and r2SCAN-D4 double relaxation",
    )

    dft_maker = MPMetaGGADoubleRelaxStaticMaker(
        relax_maker=double_relax,
        static_maker=MPMetaGGAStaticMaker(
            input_set_generator=r2scan_static_set, name="r2SCAN-D4 static"
        ),
    )

    if pre_relax_maker is not None:
        pre_relax_job = pre_relax_maker.make(structure)
        dft_relax_job = dft_maker.make(pre_relax_job.output.structure)
        flow = Flow([pre_relax_job, dft_relax_job])
    else:
        flow = dft_maker.make(structure)

    return flow


def mof_flow_revised(
    structure,
    pre_relax_maker: Maker | None = ForceFieldRelaxMaker(
        force_field_name="MATPES_R2SCAN",
        relax_kwargs={
            "fmax": 0.1,
        },
    ),
):
    nkpts = np.prod(_estimate_num_k_points_from_kspacing(structure, global_kspacing))
    user_kpoints_settings = Kpoints().as_dict()
    user_incar_settings = {}
    if nkpts < 4:
        # need 4 for tetrahedron
        user_incar_settings = {"ISMEAR": 0, "SIGMA": 0.05}

    r2scan_relax_set = MP24RelaxSet(
        user_incar_settings={
            **incar_updates_relax,
            **user_incar_settings,
        },
        user_kpoints_settings=user_kpoints_settings,
        xc_functional="r2SCAN",
        dispersion="D4",
    )

    r2scan_static_set = MP24StaticSet(
        user_incar_settings={
            **incar_updates_static,
            **user_incar_settings,
        },
        user_kpoints_settings=user_kpoints_settings,
        xc_functional="r2SCAN",
        dispersion="D4",
    )

    dft_maker = MP24DoubleRelaxStaticMaker(
        relax_maker=MPMetaGGARelaxMaker(
            input_set_generator=r2scan_relax_set, name="r2SCAN-D4 relax"
        ),
        static_maker=MPMetaGGAStaticMaker(
            input_set_generator=r2scan_static_set,
            name="r2SCAN-D4 static",
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")},
        ),
    )

    if pre_relax_maker is not None:
        pre_relax_job = pre_relax_maker.make(structure)
        dft_relax_job = dft_maker.make(pre_relax_job.output.structure)
        flow = Flow([pre_relax_job, dft_relax_job])
    else:
        flow = dft_maker.make(structure)

    return flow


def launch_jobs(
    structures="mof_single_complex.json.gz",
    aux_meta={},
):
    if isinstance(structures, str) and os.path.isfile(structures):
        structures = loadfn(structures)

    wfs = []
    for mof_id, structure in structures.items():
        job_meta = {
            "_priority": max(1.0, ceil(512.0 / len(structure))),
            "job_info": "mof_discovery_filtered",
            "mof_id": mof_id,
            "attempt": 1,
            **aux_meta.get(mof_id, {}),
        }
        job_name_prefix = f"{mof_id} "

        flow = mof_flow_revised(structure)
        flow.update_metadata(job_meta)
        flow.append_name(job_name_prefix, prepend=True)

        fw = flow_to_workflow(flow)
        fw.metadata = job_meta.copy()

        wfs.append(fw)

    return wfs


def from_zips(zip_file="./final_filtered_all_linkers_new.zip", exclude=None):
    structures = {}
    exclude = exclude or set()
    with ZipFile(zip_file, "r") as zips:
        for file_name in zips.namelist():
            if not file_name.endswith("cif"):
                continue

            if (mof_name := file_name.split("/")[-1].split(".cif")[0]) in exclude:
                continue
            struct = None
            try:
                struct = Structure.from_str(
                    zips.read(file_name).decode(encoding="utf-8"), fmt="cif"
                )
                # structures[mof_name] = struct
            except UserWarning as warn:
                print(f"{mof_name}:\n{warn}")
                continue
            structures[mof_name] = struct

    return structures


if __name__ == "__main__":
    from fireworks import LaunchPad

    lpad = LaunchPad.from_file(
        "/Users/aaronkaplan/fw_config/launchpads/my_launchpad_mpdb.yaml"
    )

    completed_fw_ids = lpad.get_fw_ids(
        {
            "spec.job_info": "mof_discovery_filtered",
            "state": "COMPLETED",
            "name": {"$regex": "r2SCAN-D4 static"},
        }
    )
    completed_mof_ids = [
        lpad.get_fw_dict_by_id(fw_id)["spec"].get("mof_id")
        for fw_id in completed_fw_ids
    ]

    all_structures = from_zips(exclude=completed_mof_ids)

    wfs = launch_jobs(structures=all_structures)
    print(len(wfs))

    # lpad.bulk_add_wfs(wfs)
