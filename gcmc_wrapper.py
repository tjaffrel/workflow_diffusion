"""
# functions for running gcmc simulations
# requires RASPA2 for all simulations (https://github.com/iRASPA/RASPA2)
# requires eGULP if atomic charges are to be assigned (https://github.com/danieleongari/egulp)
"""

# import libraries
import os
import subprocess
import re
import shutil
from pathlib import Path
from time import time
from math import cos, radians
from textwrap import dedent
from openbabel.pybel import readfile
from raspa_ase import Raspa

raspa_path = os.getenv("RASPA_PATH")
raspa_sim_path = os.getenv("RASPA_SIM_PATH")
zeo_path = os.getenv("ZEO_PATH")
egulp_path = os.getenv("EGULP_PATH")
egulp_parameter_path = os.getenv("EGULP_PARAMETER_PATH")


class gcmc_simulation:
    def __init__(
        self,
        cif_file,
        sorbates=["CO2"],
        sorbates_mol_fraction=[1],
        temperature=298,
        pressure=101325,
        rundir="./temp",
    ):
        self.sorbent = next(readfile("cif", cif_file))
        self.dim = [0, 0, 0]
        self.angle = [0, 0, 0]
        with open(cif_file) as file:
            dim_match = re.findall(r"_cell_length_.\s+\d+.\d+", file.read())
        with open(cif_file) as file:
            angle_match = re.findall(r"_cell_angle_\S+\s+\d+", file.read())
        for i in range(3):
            self.dim[i] = float(re.findall(r"\d+.\d+", dim_match[i])[0])
            self.angle[i] = float(re.findall(r"\d+", angle_match[i])[0])

        self.identifier = (
            ".".join(cif_file.split("/")[-1].split(".")[:-1])
            + "_"
            + str(time()).split(".")[1]
        )

        self.rundir = Path(rundir)
        Path(rundir).mkdir(parents=True, exist_ok=True)
        assert self.rundir.exists(), "must provide an existing rundir."
        self.sorbent_file = str(cif_file)  # noqa

        self.sorbates = sorbates
        self.sorbates_mol_fraction = [
            i / sum(sorbates_mol_fraction) for i in sorbates_mol_fraction
        ]
        self.temperature = temperature
        self.pressure = pressure

        self.block_files = None

        self.helium_void_fraction = 1.0
        self.rosenbluth_weights = [1.0 for i in range(len(sorbates))]
        self.raspa_config = None
        self.raspa_output = None

    def get_sorbate_radius(self, sorbate):
        # sorbate kinetic diameters in Angstrom
        # https://doi.org/10.1039/B802426J
        kinetic_diameter = {
            # noble gases
            "He": 2.551,
            "Ne": 2.82,
            "Ar": 3.542,
            "Kr": 3.655,
            "Xe": 4.047,
            # diatomic gases
            "H2": 2.8585,
            "D2": 2.8585,
            "N2": 3.72,
            "O2": 3.467,
            "Cl2": 4.217,
            "Br2": 4.296,
            # oxides
            "CO": 3.69,
            "CO2": 3.3,
            "NO": 3.492,
            "N2O": 3.838,
            "SO2": 4.112,
            "COS": 4.130,
            # others
            "H2O": 2.641,
            "CH4": 3.758,
            "NH3": 3.62,
            "H2S": 3.623,
        }

        try:
            return kinetic_diameter[sorbate] * 0.5
        except Exception as e:
            print("Unknown sorbate " + sorbate + ".")
            print(e)
            exit()

    def calculate_unit_cells(self, forcefield_cutoff):
        perpendicular_length = [0, 0, 0]

        for i in range(3):
            perpendicular_length[i] = self.dim[i] * abs(
                cos(radians(self.angle[i] - 90))
            )

        unit_cells = [1, 1, 1]

        for i in range(3):
            while unit_cells[i] < 2 * forcefield_cutoff / perpendicular_length[i]:
                unit_cells[i] += 1

        return unit_cells

    def write_out(self, output_path):
        with open(output_path, "w") as log_file:
            log_file.write(self.raspa_output)


# assigns charges to the atoms in the simulation file using the MEPO Qeq charge equilibration method
def calculate_mepo_qeq_charges(simulation, egulp_parameter_set="MEPO"):
    simulation.sorbent_file = str(simulation.rundir / f"{simulation.identifier}.cif")
    simulation.sorbent.write("cif", simulation.sorbent_file)
    rundir = simulation.rundir / "charges" / simulation.identifier
    rundir.mkdir(exist_ok=True, parents=True)

    config = dedent(
        """
            build_grid 0
            build_grid_from_scratch 1 none 0.25 0.25 0.25 1.0 2.0 0 0.3
            save_grid 0 grid.cube
            calculate_pot_diff 0
            calculate_pot 0 repeat.cube
            skip_everything 0
            point_charges_present 0
            include_pceq 0
            imethod 0
            """.format(**locals())
    ).strip()

    with open(rundir / "temp_config.input", "w") as file:
        file.write(config)

    # run egulp
    subprocess.run(
        [
            egulp_path,
            simulation.sorbent_file,
            os.path.join(egulp_parameter_path, egulp_parameter_set + ".param"),
            "temp_config.input",
        ],
        cwd=str(rundir),
    )

    simulation.sorbent_file = str(rundir / "charges.cif")


def run_gcmc_simulation(
    simulation,
    initialization_cycles=0,
    equilibration_cycles=2000,
    production_cycles=2000,
    forcefield="UFF",
    forcefield_cutoff=9,
    molecule_definitions="TraPPE",
    unit_cells=[0, 0, 0],
    cleanup=False,
    rewrite_raspa_input=False,
):
    shutil.copy(simulation.sorbent_file, raspa_path + "/share/raspa/structures/cif/")
    workdir = simulation.rundir / "raspa_output" / simulation.identifier
    workdir.mkdir(exist_ok=True, parents=True)

    sorbent_file = ".".join(simulation.sorbent_file.split("/")[-1].split(".")[:-1])  # noqa

    if sum(unit_cells) == 0:
        unit_cells = simulation.calculate_unit_cells(forcefield_cutoff)

    simulation.raspa_params = {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": production_cycles,
        "NumberOfInitializationCycles": initialization_cycles,
        "NumberOfEquilibrationCycles": equilibration_cycles,
        "PrintEvery": 1000,
        "Forcefield": forcefield,
        "UseChargesFromCIFFile": "yes",
        "CutOffVDW": forcefield_cutoff,
        "CutOffChargeCharge": forcefield_cutoff,
        "CutOffChargeBondDipole": forcefield_cutoff,
        "CutOffBondDipoleBondDipole": forcefield_cutoff,
        "ChargeMethod": "Ewald",
        "EwaldPrecision": 1e-6,
    }

    simulation.raspa_atoms_info = {
        "UnitCells": f"{unit_cells[0]} {unit_cells[1]} {unit_cells[2]}",
        "HeliumVoidFraction": simulation.helium_void_fraction,
        "ExternalTemperature": simulation.temperature,
        "ExternalPressure": simulation.pressure,
    }

    total_sorbates = len(simulation.sorbates)

    if total_sorbates > 1:
        identity_change_prob = 1.0
    else:
        identity_change_prob = 0.0

    sorbate_list = " ".join(str(n) for n in range(total_sorbates))

    simulation.raspa_components = []
    for i in range(total_sorbates):
        sorbate = simulation.sorbates[i]
        sorbate_mol_fraction = simulation.sorbates_mol_fraction[i]
        rosenbluth_weight = simulation.rosenbluth_weights[i]

        if simulation.block_files is None:
            block_file_line = ""
            block_flag = "no"
        else:
            block_file_line = f"\nBlockPocketsFileName          {' ' * 10} {simulation.block_files[i]}\n"
            block_flag = "yes"

        component_dict = {
            "Component": i,
            "MoleculeName": sorbate,
            "MoleculeDefinition": molecule_definitions,
            "MolFraction": sorbate_mol_fraction,
            "BlockPockets": block_flag,
            "IdealGasRosenbluthWeight": rosenbluth_weight,
            "IdentityChangeProbability": identity_change_prob,
            "NumberOfIdentityChanges": total_sorbates,
            "IdentityChangeList": sorbate_list,
            "TranslationProbability": 0.5,
            "RotationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "CreateNumberOfMolecules": 0,
        }

        if block_file_line:
            component_dict["BlockPocketsFileName"] = block_file_line.strip()

        simulation.raspa_components.append(component_dict)

    calc = Raspa(  # noqa
        components=simulation.raspa_components, parameters=simulation.raspa_params
    )

    file_list = os.listdir(str(workdir / "Output" / "System_0"))
    raspa_log = [item for item in file_list if re.match(r".*\.data", item)][0]

    with open(str(workdir / "Output" / "System_0" / raspa_log)) as log:
        simulation.raspa_output = log.read()

    if cleanup:
        shutil.rmtree(str(workdir))
