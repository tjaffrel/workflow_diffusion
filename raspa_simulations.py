import re
import os
from ase.io import read
from math import cos, radians
import random
import numpy as np
import gcmc_wrapper

from raspa_ase.calculator import Raspa

workdir = "/home/theoj/project/diffusion/workflow"
cif_file = "MFI-SI.cif"
atoms = read(cif_file)
forcefield = "Dubbeldam2007FlexibleIRMOF-1"
forcefield_cutoff = 9


output_dir = os.path.join(workdir, "Output", "System_0")
file_list = os.listdir(output_dir)
raspa_log = [item for item in file_list if re.match(r".*\.data", item)][0]

with open(os.path.join(output_dir, raspa_log)) as log:
    simulation = log.read()


def extract_raspa_output(raspa_output):
    final_loading_section = re.findall(
        r"Number of molecules:\n=+[^=]*(?=)", raspa_output
    )[0]
    enthalpy_of_adsorption_section = re.findall(
        r"Enthalpy of adsorption:\n={2,}\n(.+?)\n={2,}", raspa_output, re.DOTALL
    )[0]

    subsection = re.findall(
        r"Component \d \[CO2\].*?(?=Component|\Z)", final_loading_section, re.DOTALL
    )[0]
    adsorbed = float(
        re.findall(
            r"(?<=Average loading absolute \[mol/kg framework\])\s*\d*\.\d*",
            subsection,
        )[0]
    )

    enthalpy_subsection = re.findall(
        r"Total enthalpy of adsorption.*?(?=Q=-H|\Z)",
        enthalpy_of_adsorption_section,
        re.DOTALL,
    )[0]
    # conversion to kcal per mol
    enthalpy_of_adsorption = (
        float(re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", enthalpy_subsection)[0]) * 0.239
    )
    heat_of_adsorption = -1 * enthalpy_of_adsorption

    return adsorbed, heat_of_adsorption


def calculate_unit_cells(forcefield_cutoff, cif_file):
    perpendicular_length = [0, 0, 0]
    dim = [0, 0, 0]
    angle = [0, 0, 0]
    with open(cif_file) as file:
        dim_match = re.findall(r"_cell_length_.\s+\d+.\d+", file.read())
    with open(cif_file) as file:
        angle_match = re.findall(r"_cell_angle_\S+\s+\d+", file.read())
    for i in range(3):
        dim[i] = float(re.findall(r"\d+.\d+", dim_match[i])[0])
        angle[i] = float(re.findall(r"\d+", angle_match[i])[0])

    for i in range(3):
        perpendicular_length[i] = dim[i] * abs(cos(radians(angle[i] - 90)))

    unit_cells = [1, 1, 1]

    for i in range(3):
        while unit_cells[i] < 2 * forcefield_cutoff / perpendicular_length[i]:
            unit_cells[i] += 1

    return unit_cells


def working_capacity_vacuum_swing(
    cif_file, calc_charges=True, rundir="./temp", rewrite_raspa_input=False
):
    random.seed(4)
    np.random.seed(4)
    # adsorption conditions
    adsorbed = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["CO2"],
        sorbates_mol_fraction=[0.15, 0.85],
        temperature=298,
        pressure=100000,  # 1 bar
        rundir=rundir,
    )

    if calc_charges:
        gcmc_wrapper.calculate_mepo_qeq_charges(adsorbed)
    gcmc_wrapper.run_gcmc_simulation(
        adsorbed,
        rewrite_raspa_input=rewrite_raspa_input,
    )

    (
        adsorbed_CO2,
        heat_of_adsorption_CO2_298,
    ) = extract_raspa_output(adsorbed.raspa_output, has_N2=True)

    # desorption conditions
    residual = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["CO2"],
        sorbates_mol_fraction=[1],
        temperature=363,  # 363,
        pressure=10000,  # 10000 # 0.1 bar
        rundir=rundir,
    )

    if calc_charges:
        gcmc_wrapper.calculate_mepo_qeq_charges(residual)
    gcmc_wrapper.run_gcmc_simulation(
        residual,
        rewrite_raspa_input=rewrite_raspa_input,
    )

    residual_CO2, heat_of_adsorption_CO2_363 = extract_raspa_output(
        residual.raspa_output, has_N2=False
    )

    output = {
        "file": str(cif_file),
        "working_capacity_vacuum_swing": adsorbed_CO2 - residual_CO2,
        "CO2_uptake_P0.15bar_T298K": adsorbed_CO2,
        "CO2_uptake_P0.10bar_T363K": residual_CO2,
        "CO2_heat_of_adsorption_P0.15bar_T298K": heat_of_adsorption_CO2_298,
        "CO2_heat_of_adsorption_P0.10bar_T363K": heat_of_adsorption_CO2_363,
    }

    return output


def run_or_fail(cif_path):
    try:
        return working_capacity_vacuum_swing(cif_path)
    except Exception as e:
        print(e)
        return None


unit_cells = calculate_unit_cells(forcefield_cutoff, cif_file)

atoms.info = {
    "UnitCells": unit_cells,
    "ExternalTemperature": 300.0,
    "FrameworkName": "IRMOF-1",
    "FlexibleFramework": "no",
}
parameters = {
    "SimulationType": "Minimization",
    "NumberOfCycles": 1,
    "PrintEvery": 1,
    "MaximumNumberOfMinimizationSteps": 1000,
    "Forcefield": forcefield,
    "RMSGradientTolerance": 1e-4,
    "MaxGradientTolerance": 1e-4,
    "Ensemble": "NVT",
    "CutOffVDW": forcefield_cutoff,
    "CutOffChargeCharge": forcefield_cutoff,
    "CutOffChargeBondDipole": forcefield_cutoff,
    "CutOffBondDipoleBondDipole": forcefield_cutoff,
    "ChargeMethod": "Ewald",
    "EwaldPrecision": 1e-6,
    "InternalFrameworkLennardJonesInteractions": "yes",
}

print(parameters, atoms.info)
calc = Raspa(parameters=parameters)  # noqa
