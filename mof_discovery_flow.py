from __future__ import annotations
from atomate2.forcefields.jobs import MACERelaxMaker
from dataclasses import dataclass, field
from quacc.recipes.dftb.core import relax_job as gfn_xtb_relax_job
from quacc.runners.ase import run_calc
from quacc.schemas._aliases.ase import RunSchema
from quacc.schemas.ase import summarize_run

from jobflow import Flow, Maker, job
from pathlib import Path
from raspa_ase import Raspa

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from jobflow import Job
    from pymatgen.core import Structure
    from typing import Any


@job
def run_raspa(
    atoms: Atoms,
    raspa_calculator_kwargs: dict | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> RunSchema:
    calc = Raspa(**raspa_calculator_kwargs)

    atoms.calc = calc

    # TODO:
    # how do we structure input for RASPA?
    # where does RASPA store output geometries / name of this file? Needed for `geom_file`
    # which files do we copy over for RASPA input / output? Needed for `copy_files`
    final_atoms = run_calc(atoms, geom_file=None, copy_files=None)

    return summarize_run(final_atoms, atoms, additional_fields=additional_fields)


@dataclass
class MofDiscoveryFLow(Maker):
    name: str = "MOF discovery workflow"
    ff_relax_maker: Maker = field(
        default_factory=lambda: MACERelaxMaker(
            calculator_kwargs={
                "model": "small",
                "default_dtype": "float32",
                "dispersion": True,
            },
            steps=200,
        )
    )
    tb_relax_maker_kwargs: dict | None = None
    run_raspa: bool = False
    raspa_kwargs: dict | None = None

    def make(
        self, structure: Structure | str | Path, job_meta: dict | None = None
    ) -> Flow:
        if isinstance(structure, (str, Path)):
            structure = Structure.from_file(structure)

        job_meta = job_meta or {}

        jobs: list[Job] = []
        mace_relax_job = self.ff_relax_maker.make(structure)
        mace_relax_job.metadata = {"job_type": "mace-relax", **job_meta}
        jobs += [mace_relax_job]

        self.tb_relax_maker_kwargs = self.tb_relax_maker_kwargs or {}
        tb_relax_job = gfn_xtb_relax_job(
            mace_relax_job.output.output.structure.to_ase_atoms(),
            **self.tb_relax_maker_kwargs,
        )
        tb_relax_job.metadata = {"job_type": "gfn-xtb-relax", **job_meta}
        jobs += [tb_relax_job]
        output = tb_relax_job.output

        if self.run_raspa:
            self.raspa_kwargs = self.raspa_kwargs or {}
            raspa_job = run_raspa(tb_relax_job.output["atoms"])
            raspa_job.metadata = {"job_type": "raspa", **job_meta}
            jobs += [raspa_job]
            output = raspa_job.output

        return Flow(jobs, output=output)
