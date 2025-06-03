"""Script to start training a study."""

####################################
# Packages
####################################
import os
import shutil
import argparse
import pandas as pd
import subprocess
from pathlib import Path

from yoke.helpers import cli, strings, create_slurm_files


####################################
# Process Hyperparameters
####################################
parser = argparse.ArgumentParser(
    prog="HARNESS START", description="Starts execution of training harness"
)
parser = cli.add_default_args(parser)
args = parser.parse_args()

training_START_input = "./training_START.input"
training_input_tmpl = "./training_input.tmpl"

if args.submissionType.lower() == "slurm":
    training_START_slurm = "./training_START.slurm"
    training_slurm_tmpl = "./training_slurm.tmpl"

if args.submissionType.lower() == "flux":
    training_START_flux = "./training_START.flux"
    training_flux_tmpl = "./training_flux.tmpl"

training_json = "./slurm_config.json"

slurm_tmpl_data = None
if os.path.exists(training_json):
    slrm_obj = create_slurm_files.MkSlurm(config_path=training_json)
    slurm_tmpl_data = slrm_obj.generateSlurm()

# List of files to copy
with open(args.cpFile) as cp_text_file:
    cp_file_list = [line.strip() for line in cp_text_file]

# Process Hyperparameters File
studyDF = pd.read_csv(
    args.csv, sep=",", header=0, index_col=0, comment="#", engine="python"
)
varnames = studyDF.columns.values
idxlist = studyDF.index.values

# Save Hyperparameters to list of dictionaries
studylist = []
for i in idxlist:
    studydict = {}
    studydict["studyIDX"] = int(i)

    for var in varnames:
        studydict[var] = studyDF.loc[i, var]

    studylist.append(studydict)

####################################
# Run Studies
####################################
# Iterate Through Dictionary List to Run Studies
for k, study in enumerate(studylist):
    # Make Study Directory
    studydirname = args.rundir + "/study_{:03d}".format(study["studyIDX"])

    if not os.path.exists(studydirname):
        os.makedirs(studydirname)

    # Make new training_input.tmpl file
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    training_input_data = strings.replace_keys(study, training_input_data)
    training_input_filepath = os.path.join(studydirname, "training_input.tmpl")

    with open(training_input_filepath, "w") as f:
        f.write(training_input_data)

    # Make new training_START.input file
    with open(training_START_input) as f:
        START_input_data = f.read()

    START_input_data = strings.replace_keys(study, START_input_data)
    START_input_name = "study{:03d}_START.input".format(study["studyIDX"])
    START_input_filepath = os.path.join(studydirname, START_input_name)

    with open(START_input_filepath, "w") as f:
        f.write(START_input_data)

    # Create submission scripts based on submission system.
    if args.submissionType.lower() == "slurm":
        # Make new training_slurm.tmpl file
        if slurm_tmpl_data is None:
            with open(training_slurm_tmpl) as f:
                training_slurm_data = f.read()

        else:
            training_slurm_data = slurm_tmpl_data

        training_slurm_data = strings.replace_keys(study, training_slurm_data)
        training_slurm_filepath = os.path.join(studydirname, "training_slurm.tmpl")

        with open(training_slurm_filepath, "w") as f:
            f.write(training_slurm_data)

        if slurm_tmpl_data is None:
            # Make a new training_START.slurm file
            with open(training_START_slurm) as f:
                START_slurm_data = f.read()

        if slurm_tmpl_data is not None:
            START_slurm_data = strings.replace_keys(study, slurm_tmpl_data).replace(
                "<epochIDX>", "0001"
            )

        else:
            START_slurm_data = strings.replace_keys(study, START_slurm_data)

        START_slurm_name = "study{:03d}_START.slurm".format(study["studyIDX"])
        START_slurm_filepath = os.path.join(studydirname, START_slurm_name)

        with open(START_slurm_filepath, "w") as f:
            f.write(START_slurm_data)

    training_START_flux = "./training_START.flux"
    training_flux_tmpl = "./training_flux.tmpl"
    if args.submissionType.lower() == "flux":
        # Make new training_flux.tmpl file
        with open(training_flux_tmpl) as f:
            training_flux_data = f.read()

        training_flux_data = strings.replace_keys(study, training_flux_data)
        training_flux_filepath = os.path.join(studydirname, "training_flux.tmpl")

        with open(training_flux_filepath, "w") as f:
            f.write(training_flux_data)

        # Make a new training_START.slurm file
        with open(training_START_flux) as f:
            START_flux_data = f.read()

        START_flux_data = strings.replace_keys(study, START_flux_data)

        START_flux_name = "study{:03d}_START.flux".format(study["studyIDX"])
        START_flux_filepath = os.path.join(studydirname, START_flux_name)

        with open(START_flux_filepath, "w") as f:
            f.write(START_flux_data)

    # Copy files to study directory from list
    for f in cp_file_list:
        shutil.copy(f, studydirname)

    # Submit a job with the appropriate submission type
    if args.submissionType.lower() == "slurm":
        slurm_cmd = (
            f"cd {studydirname}; sbatch {START_slurm_name}; "
            f"cd {os.path.dirname(__file__)}"
        )
        os.system(slurm_cmd)

    if args.submissionType.lower() == "flux":
        flux_cmd = (
            f"cd {studydirname}; flux batch {START_slurm_name}; "
            f"cd {os.path.dirname(__file__)}"
        )
        os.system(flux_cmd)

    elif args.submissionType.lower() == "shell":
        wrapper = Path(__file__).parent / "run_study.sh"
        subprocess.run(
            [str(wrapper), study["train_script"], START_input_name],
            cwd=studydirname,
            check=True,
        )

    elif args.submissionType.lower() == "batch":
        harness_dir = Path(args.csv).parent.resolve()
        wrapper = harness_dir / "run_study.bat"
        if not wrapper.exists():
            raise FileNotFoundError(f"Cannot find batch wrapper at {wrapper!r}")

        abs_wrapper = str(wrapper)

        subprocess.run(
            ["cmd.exe", "/c", str(wrapper), study["train_script"], START_input_name],
            cwd=studydirname,
            check=True,
        )

    else:
        raise ValueError(f"Unknown submission type: {args.submissionType!r}")
