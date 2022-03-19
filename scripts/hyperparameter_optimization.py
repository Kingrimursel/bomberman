#!/bin/python3

## before using this script:
# - create config file accordingly
# - load alpha, gamma, epsilon from imported config file (create __init__ überall)
# - implement new logging
# - übernehmen: truly_training
# - übernehmen: keeping track of other scores
# - übernehmen: deterministic next step
# - best way: check changelog in github

import os
import sys
import pickle
import subprocess
import argparse

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


model_name       = "my-saved-model.pt"

num_of_test_sessions = 1  # 10

step_number_alpha = 2  # 20
step_number_gamma = 2

lower_bound_alpha = 0.25  # lower bound excluded
upper_bound_alpha = 0.75 # upper bound included

lower_bound_gamma = 0.5  # lower bound excluded
upper_bound_gamma = 1  # upper bound included

step_size_alpha = (upper_bound_alpha - lower_bound_alpha)/step_number_alpha
step_size_gamma = (upper_bound_gamma - lower_bound_gamma)/step_number_gamma


mean_placements = np.empty((step_number_alpha, step_number_gamma))
mean_scores     = np.empty((step_number_alpha, step_number_gamma))



def main():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-a", "--agent", help = "agent name")
    # Read arguments from command line
    args = parser.parse_args()

    agent_name   = args.agent

    if not agent_name:
        print("PLEASE PROVIDE AN AGENT NAME")
        return


    test_command = f"python3 main.py play --my-agent {agent_name} --train 1 --no-gui --scenario coin-heaven"


    dt = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    base_dir = Path(f"../agent_code/{agent_name}")


    # move model
    if os.path.exists(os.path.join(base_dir, model_name)):
        os.rename(os.path.join(base_dir, model_name), os.path.join(base_dir, model_name + "_old"))


    ## loop over all combinations of alpha and gamma values. i = alpha, j = gamma
    counter = 1
    limit_exceeded = False
    for i in range(1, step_number_alpha + 1):

        if limit_exceeded:
            break

        for j in range(1, step_number_gamma + 1):

            subdir = f"{dt}/i={i},j={j}"

            analysis_directory = Path(os.path.join(base_dir, "logs/analysis", subdir))
            history_path       = Path(os.path.join(analysis_directory, "history.npy"))
            placement_path     = Path(os.path.join(analysis_directory, "placement.npy"))
            config_path         = Path(os.path.join(base_dir, "config.py"))

            ## training
            # set starting val for random prob
            alpha = i*step_size_alpha + lower_bound_alpha
            gamma = j*step_size_gamma + lower_bound_gamma

            if alpha > upper_bound_alpha or gamma > upper_bound_gamma:
                limit_exceeded = True

            if limit_exceeded:
                break

            update_var("ALPHA", alpha, config_path)
            update_var("GAMMA", gamma, config_path)
            update_var("TRULY_TRAIN", True, config_path)

            print(f"{color.RED}ITERATION {counter} OF {step_number_alpha*step_number_gamma} {color.NC}")

            # actually training
            if counter == 1:
                subprocess.run([f"python3 train.py -a {agent_name} -d {subdir} -dp -c"], shell=True)
            else:
                subprocess.run([f"python3 train.py -a {agent_name} -d {subdir} -dp"], shell=True)


            # since we are now testing, not training
            clear_data(history_path, placement_path)
            update_var("TRULY_TRAIN", False, config_path)


            # testing
            for session_nr in range(num_of_test_sessions):
                print(f"{color.BLUE}testing Nr. {session_nr + 1} of {num_of_test_sessions}: {color.NC} i={i}, j={j}, alpha={alpha}, gamma={gamma}")
                os.chdir("..")
                subprocess.run([test_command], shell=True)
                os.chdir("scripts")
                # evaluate logs
                subprocess.run([f"python3 analyze_logs.py -a {agent_name} -d {subdir}"], shell=True)


            placements, scores = calculate_ratings(placement_path)

            placement_mean = np.mean(np.array(placements))
            score_mean     = np.mean(np.array(scores))
            mean_placements[i-1, j-1] = placement_mean
            mean_scores[i-1, j-1]     = score_mean

            # move model to directory
            os.rename(os.path.join(base_dir, model_name), os.path.join(analysis_directory.parents[0], model_name))

            counter += 1


    ## plot the winrate surface
    x = np.linspace(lower_bound_alpha + step_size_alpha, upper_bound_alpha, step_number_alpha)
    y = np.linspace(lower_bound_gamma + step_size_gamma, upper_bound_gamma, step_number_gamma)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot_wireframe(X, Y, mean_placements, color="green")
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'average placement')


    i_opt, j_opt = find_optimal_settings(mean_placements, mean_scores)

    alpha_opt = (i_opt + 1)*step_size_alpha + lower_bound_alpha
    gamma_opt = (j_opt + 1)*step_size_gamma + lower_bound_gamma

    print(f"{color.GREEN}OPTIMIZATION FINISHED. Optimal parameters: i={i_opt+1}, alpha={alpha_opt}, j={j_opt+1}, gamma={gamma_opt}{color.NC}")


    update_var("ALPHA", alpha_opt, config_path)
    update_var("GAMMA", gamma_opt, config_path)
    update_var("TRULY_TRAIN", True, config_path)

    print(mean_placements)
    print(mean_scores)

    plt.savefig(os.path.join(analysis_directory.parents[0], "optimization_landscape.png"))
    plt.savefig(f"output/optimization_landscape_{agent_name}.png")

    with open(f'output/optimization_landscape_{agent_name}.fig.pickle', 'wb') as file:
        pickle.dump(fig, file)

    with open(os.path.join(analysis_directory.parents[0], "optimization_landscape.fig.pickle"), 'wb') as file:
        pickle.dump(fig, file)

    # figx = pickle.load(open('optimization_landscape.fig.pickle', 'rb'))
    # plt.show()

    with open(os.path.join(analysis_directory.parents[0], "info.txt"), 'w') as file:
        file.write(
            f"\
            step_number_alpha={step_number_alpha}\n\
            step_number_gamma={step_number_gamma}\n\
            num_of_test_sessions={num_of_test_sessions}\n\
            lower_bound_alpha={lower_bound_alpha}\n\
            upper_bound_alpha={upper_bound_alpha}\n\
            lower_bound_gamma={lower_bound_gamma}\n\
            upper_bound_gamma={upper_bound_gamma}\n\
            mean_placements={mean_placements}\n\
            mean_score={mean_scores}\n\
            alpha_opt={alpha_opt}\n\
            gamma_opt={gamma_opt}\n\
            i_opt={i_opt+1}\n\
            j_opt={j_opt+1}\n\n\
            To create the image output: cd into scripts/ and run -/process_log_data.py -a [agent_name] -d [datetime/i=i,j=j]"
            )


def clear_data(history_path, placement_path):
    history_path.unlink(missing_ok=True)
    placement_path.unlink(missing_ok=True)


def calculate_ratings(datapath):
    with open(datapath, "rb") as file:
        history = pickle.load(file)

        placements = []
        scores     = []

        for gen in history:
            placements.extend(np.array(history[gen])[:, 0])
            scores.extend(np.array(history[gen])[:, 1])


    return placements, scores

def update_var(name, value, configpath):
    newcontent = ""


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


def find_optimal_settings(mean_placements, mean_scores):
    """
    Find optimal settings for hyperparameters by first minimizing the mean placement and in case of a tie maximizing the
    mean score.

    INPUT: the tables
    """

    # TODO: make this more robust by not checking for equality but similarity

    w, h = mean_placements.shape


    lowest_mean = 100
    lowest_mean_indices = []


    for i in range(w):
        for j in range(h):

            if mean_placements[i, j] == lowest_mean:
                lowest_mean_indices.append((i, j))

            if mean_placements[i, j] < lowest_mean:
                lowest_mean = mean_placements[i, j]
                lowest_mean_indices = [(i, j)]


    highest_score = 0
    optimal_indices = None

    for i, j in lowest_mean_indices:
        if mean_scores[i, j] > highest_score:
            highest_score = mean_scores[i, j]
            optimal_indices = (i, j)


    return optimal_indices





if __name__ == "__main__":
    main()
