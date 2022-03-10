#!/bin/python3

## before using this script:
# - create config file accordingly
# - load alpha, gamma, random_prob from imported config file (create __init__ überall)
# - implement new logging

import os
import sys
import pickle
import subprocess

from pathlib import Path

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


agent_name = "own_coin"
training_command = f"python3 main.py play --my-agent {agent_name} --train 1 --no-gui --scenario coin-heaven"
test_command = f"python3 main.py play --my-agent {agent_name} --no-gui --train 1 --scenario coin-heaven"

# lower random probability after ... training sessions (exploration phase)
lower_random_prob_after = 1  # 10
# lower random probability to...
lower_random_prob_to    = 0.1
# the random probability we are starting the training with
starting_random_prob     = 0.2

num_of_training_sessions = 2  # 50
num_of_test_sessions  = 2  # 10

step_number_alpha = 2  # 20
step_number_gamma = 2
step_size_alpha = 1/step_number_alpha
step_size_gamma = 1/step_number_gamma
mean_placements = np.empty((step_number_alpha, step_number_gamma))


# TODO Elias soll nochmal eine gute training-Strategie nennen

# TODO: was anderes im training zählen! nicht nur wie lange er überlebt, sondern den platz
# TODO: momentan wird auch im testing trainiert. Gibt es eine kluge methode, wie man im trainingsmodus nicht trainiert?
# JA, EINFACH Q-TABLE NICHT UPDATEN! add config boolean

def main():

    history_path = Path("../agent_code/{}/logs/analysis/history.npy".format(agent_name))
    placement_path = Path("../agent_code/{}/logs/analysis/placement.npy".format(agent_name))

    history_path.unlink(missing_ok=True)
    placement_path.unlink(missing_ok=True)

    os.chdir("..")

    
    ## loop over all combinations of alpha and gamma values. i = alpha, j = gamma
    counter = 1
    for i in range(1, step_number_alpha + 1):
        for j in range(1, step_number_gamma + 1):
            
            ## training
            # set starting value for random prob
            update_var("RANDOM_PROB", starting_random_prob)
            update_var("ALPHA", i*step_size_alpha)
            update_var("GAMMA", j*step_size_gamma)
            update_var("TRULY_TRAIN", True)

            print(f"{color.RED}ITERATION {counter} OF {step_number_alpha*step_number_gamma} {color.NC}")

            # do the training sessions
            for session_nr in range(num_of_training_sessions):
                print(f"{color.PURPLE}training Nr. {session_nr + 1} of {num_of_training_sessions}:{color.NC} i={i}, j={j}, alpha={i*step_size_alpha}, gamma={j*step_size_gamma}")
                # lower random prob (exploration phase is over)
                if session_nr == lower_random_prob_after:
                    update_var("RANDOM_PROB", lower_random_prob_to)

                subprocess.run([training_command], shell=True)
            


            update_var("TRULY_TRAIN", False)
            # testing
            for session_nr in range(num_of_test_sessions):
                print(f"{color.BLUE}testing Nr. {session_nr + 1} of {num_of_test_sessions}: {color.NC} i={i}, j={j}, alpha={i*step_size_alpha}, gamma={j*step_size_gamma}")
                subprocess.run([test_command], shell=True)
                # evaluate logs
                os.chdir("scripts")
                subprocess.run([f"./analyze_logs.py -a {agent_name}"], shell=True)
                os.chdir("..")
            

            placements = calculate_placements()

            placement_mean = np.mean(np.array(placements))
            mean_placements[i-1, j-1] = placement_mean

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

    print(f"{color.GREEN}OPTIMIZATION FINISHED. Optimal parameters: alpha={alpha_opt}, gamma={gamma_opt}{color.NC}")


    update_var("ALPHA", alpha_opt)
    update_var("GAMMA", gamma_opt)
    update_var("TRULY_TRAIN", True)
    os.chdir("scripts")

    print(mean_placements)

    plt.savefig(f"output/optimization_landscape_{agent_name}.png")
    with open(f'output/optimization_landscape_{agent_name}.fig.pickle', 'wb') as file:
        pickle.dump(fig, file)
    # figx = pickle.load(open('optimization_landscape_own_coin.fig.pickle', 'rb'))
    # plt.show()

def calculate_placements():
    datapath = Path(f"agent_code/{agent_name}/logs/analysis/placement.npy")

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