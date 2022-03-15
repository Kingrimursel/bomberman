#!/bin/python3



import os
import sys
import pickle
import subprocess
import argparse

from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


class color:
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    NC='\033[0m'


model_name = "my-saved-model.pt"

# lower random probability after ... training sessions (exploration phase)
lower_random_prob_after = 5  # 10
# lower random probability to...
lower_random_prob_to    = 0.1
# the random probability we are starting the training with
starting_random_prob    = 0.2

# train deterministically for ... rounds
deterministic_for = 5

num_of_training_sessions = 20  # 50



def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-a", "--agent", help = "agent name")
    parser.add_argument("-c", "--clear", help = "clear old data", action="store_true")
    parser.add_argument("-d", "--directory", help = "subdirectory to save images in")
    parser.add_argument("-dp", "--dontprocess", action="store_true", help = "process logs")

    # Read arguments from command line
    args = parser.parse_args()

    agent_name   = args.agent
    clear        = args.clear
    subdir       = args.directory
    process_logs = not args.dontprocess

    if not agent_name:
        print("PLEASE PROVIDE AN AGENT NAME")
        return

    training_command = f"python3 main.py play --my-agent {agent_name} --train 1 --no-gui --scenario coin-heaven"


    base_dir = Path(f"../agent_code/{agent_name}") 

    if subdir:
        analysis_directory = Path(os.path.join(base_dir, "logs/analysis", subdir))
    else:
        analysis_directory = Path(os.path.join(base_dir, "logs/analysis"))


    history_path       = Path(os.path.join(analysis_directory, "history.npy"))
    placement_path     = Path(os.path.join(analysis_directory, "placement.npy"))


    # move old model
    if clear:
        os.rename(os.path.join(base_dir, model_name), os.path.join(base_dir, model_name + "_old"))


    os.chdir("..")   
   
    update_var("RANDOM_PROB", starting_random_prob, agent_name)
    update_var("DETERMINISTIC", True, agent_name)

    clear_data(history_path, placement_path)

    # do the training sessions
    counter = 1
    for session_nr in range(num_of_training_sessions):
        print(f"{color.PURPLE}training Nr. {session_nr + 1} of {num_of_training_sessions}:{color.NC}")

        # make training not deterministic
        if session_nr == deterministic_for:
            update_var("DETERMINISTIC", False, agent_name)

        # lower random prob (exploration phase is over)
        if session_nr == deterministic_for + lower_random_prob_after:
            update_var("RANDOM_PROB", lower_random_prob_to, agent_name)

        subprocess.run([training_command], shell=True)

        # analyze the logs
        os.chdir("scripts")
        if clear and counter == 1:
            subprocess.run([f"./analyze_logs.py -a {agent_name} -d {subdir} -c"], shell=True)
        else:
            subprocess.run([f"./analyze_logs.py -a {agent_name} -d {subdir}"], shell=True)

        os.chdir("..")
        counter += 1


    # sicherheitshalber, falls loop empty war:
    update_var("DETERMINISTIC", False, agent_name)
    update_var("RANDOM_PROB", lower_random_prob_to, agent_name)

    os.chdir("scripts")

    if process_logs:
        subprocess.run([f"./process_log_data.py -a {agent_name} -d {subdir}"], shell=True)



def clear_data(history_path, placement_path):
    history_path.unlink(missing_ok=True)
    placement_path.unlink(missing_ok=True)


def update_var(name, value, agent_name):
    newcontent = ""

    configpath = Path(f"agent_code/{agent_name}/config.py")

    with open(configpath, "r") as file:
        for line in file:
            els = line.split(" ")
            thisname = els[0]
            if thisname == name:
                els[-1] = str(value) + "\n"
                line = " ".join(els)

            newcontent += line

    with open(configpath, "w") as file:
        file.write(newcontent)


if __name__ == "__main__":
    main()