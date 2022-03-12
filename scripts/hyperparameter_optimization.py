#!/bin/python3

## before using this script:
# - create config file accordingly
# - load alpha, gamma, random_prob from imported config file (create __init__ überall)
# - implement new logging
# - übernehmen: truly_training
# - übernehmen: keeping track of other scores
# - best way: check changelog in github

import os
import sys
import pickle
import subprocess

from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt



sys.path.append(os.path.abspath(".."))

from agent_code.own_coin import config


class color:
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    NC='\033[0m'


agent_name       = "own_coin"
training_command = f"python3 main.py play --my-agent {agent_name} --train 1 --no-gui --scenario coin-heaven"
test_command     = f"python3 main.py play --my-agent {agent_name} --no-gui --train 1 --scenario coin-heaven"
model_name       = "my-saved-model.pt"

# lower random probability after ... training sessions (exploration phase)
lower_random_prob_after = 1  # 10
# lower random probability to...
lower_random_prob_to    = 0.1
# the random probability we are starting the training with
starting_random_prob     = 0.2

num_of_training_sessions = 1  # 50
num_of_test_sessions     = 1  # 10

step_number_alpha = 1  # 20
step_number_gamma = 1
step_size_alpha = 1/step_number_alpha
step_size_gamma = 1/step_number_gamma
mean_placements = np.empty((step_number_alpha, step_number_gamma))


# TODO Elias soll nochmal eine gute training-Strategie nennen

def main():
    print(f"{color.RED}SET UP A NEW MODEL!?!?{color.NC}")
    print(f"{color.RED}SET UP A NEW MODEL!?!?{color.NC}")
    print(f"{color.RED}SET UP A NEW MODEL!?!?{color.NC}")
    print(f"{color.RED}SET UP A NEW MODEL!?!?{color.NC}\n\n")

    dt = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    base_dir = Path(f"../agent_code/{agent_name}")

    os.chdir("..")
    
    ## loop over all combinations of alpha and gamma values. i = alpha, j = gamma
    counter = 1
    for i in range(1, step_number_alpha + 1):
        for j in range(1, step_number_gamma + 1):
            
            subdir = f"{dt}/i={i},j={j}"

            analysis_directory = Path(os.path.join(base_dir, "logs/analysis", subdir))
            history_path       = Path(os.path.join(analysis_directory, "history.npy"))
            placement_path     = Path(os.path.join(analysis_directory, "placement.npy"))
        
            clear_data(history_path, placement_path)


            ## training
            # set starting value for random prob
            alpha = i*step_size_alpha
            gamma = j*step_size_gamma

            update_var("RANDOM_PROB", starting_random_prob)
            update_var("ALPHA", alpha)
            update_var("GAMMA", gamma)
            update_var("TRULY_TRAIN", True)

            print(f"{color.RED}ITERATION {counter} OF {step_number_alpha*step_number_gamma} {color.NC}")

            # TODO: alternating optimization. Maybe listen for key to stop loop?

            # clear data since we will now evaluate testing, not training
            clear_data(history_path, placement_path)
            # do the training sessions
            for session_nr in range(num_of_training_sessions):
                print(f"{color.PURPLE}training Nr. {session_nr + 1} of {num_of_training_sessions}:{color.NC} i={i}, j={j}, alpha={alpha}, gamma={gamma}")
                # lower random prob (exploration phase is over)
                if session_nr == lower_random_prob_after:
                    update_var("RANDOM_PROB", lower_random_prob_to)

                subprocess.run([training_command], shell=True)


                # analyze the logs
                os.chdir("scripts")
                subprocess.run([f"./analyze_logs.py -a {agent_name} -d {subdir}"], shell=True)
                os.chdir("..")

            # os.chdir("scripts")
            # subprocess.run([f"./process_log_data.py -a {agent_name} -d {subdir}"], shell=True)
            # os.chdir("..")


            # since we are now testing, not training
            clear_data(history_path, placement_path)
            update_var("TRULY_TRAIN", False)
            # testing
            for session_nr in range(num_of_test_sessions):
                print(f"{color.BLUE}testing Nr. {session_nr + 1} of {num_of_test_sessions}: {color.NC} i={i}, j={j}, alpha={alpha}, gamma={gamma}")
                subprocess.run([test_command], shell=True)
                # evaluate logs
                os.chdir("scripts")
                subprocess.run([f"./analyze_logs.py -a {agent_name} -d {subdir}"], shell=True)
                os.chdir("..")
            

            os.chdir("scripts")
            placements = calculate_placements(placement_path)
            os.chdir("..")

            placement_mean = np.mean(np.array(placements))
            mean_placements[i-1, j-1] = placement_mean

            # move model to directory
            os.chdir("scripts")
            os.rename(os.path.join(base_dir, model_name), os.path.join(analysis_directory.parents[0], model_name))
            os.chdir("..")

            counter += 1


    ## plot the winrate surface
    x = np.linspace(0, 1, step_number_alpha)
    y = np.linspace(0, 1, step_number_gamma)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot_wireframe(X, Y, mean_placements, color="green")
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'average placement')


    argmin_vals_columns = np.argmin(mean_placements, axis=0)
    argmin_val_column = np.argmin(mean_placements[argmin_vals_columns, np.arange(step_number_gamma)])
    argmin_val_row    = argmin_vals_columns[argmin_val_column]

    alpha_opt = (argmin_val_row+1)*step_size_alpha
    gamma_opt = (argmin_val_column+1)*step_size_gamma

    print(f"{color.GREEN}OPTIMIZATION FINISHED. Optimal parameters: i={argmin_val_row+1}, alpha={alpha_opt}, j={argmin_val_column+1}, gamma={gamma_opt}{color.NC}")


    update_var("ALPHA", alpha_opt)
    update_var("GAMMA", gamma_opt)
    update_var("TRULY_TRAIN", True)
    os.chdir("scripts")

    print(mean_placements)

    plt.savefig(os.path.join(analysis_directory.parents[0], "optimization_landscape.png"))
    plt.savefig(f"output/optimization_landscape_{agent_name}.png")

    with open(f'output/optimization_landscape_{agent_name}.fig.pickle', 'wb') as file:
        pickle.dump(fig, file)

    with open(os.path.join(analysis_directory.parents[0], "optimization_landscape.fig.pickle"), 'wb') as file:
        pickle.dump(fig, file)

    with open(os.path.join(analysis_directory.parents[0], "info.txt"), 'w') as file:
        file.write(
            f"lower_random_prob_after={lower_random_prob_after}\n\
lower_random_prob_to={lower_random_prob_to}\n\
starting_random_prob={starting_random_prob}\n\
step_number_alpha={step_number_alpha}\n\
step_number_gamma={step_number_gamma}\n\
num_of_training_sessions={num_of_training_sessions}\n\
num_of_test_sessions={num_of_test_sessions}\n\
mean_placements={mean_placements}\n\
alpha_opt={alpha_opt}\n\
gamma_opt={gamma_opt}\n\
i_opt={argmin_val_row+1}\n\
j_opt={argmin_val_column+1}\n\n\
To create the image output: cd into scripts/ and run -/process_log_data.py -a [agent_name] -d [datetime/i=i,j=j]"
            )

    # figx = pickle.load(open('optimization_landscape_own_coin.fig.pickle', 'rb'))
    # plt.show()


def clear_data(history_path, placement_path):
    history_path.unlink(missing_ok=True)
    placement_path.unlink(missing_ok=True)


def calculate_placements(datapath):
    with open(datapath, "rb") as file:
        history = pickle.load(file)

        placements = []

        for gen in history:
            placements.extend(history[gen])


    return placements

def update_var(name, value):
    newcontent = ""

    configpath = Path(f"agent_code/{agent_name}/config.py")

    with open(configpath, "r") as file:
        #print(file.readlines())
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