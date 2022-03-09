#!/bin/python3

import argparse
import pickle
import shutil
import os

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt



def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
     
    # Adding optional argument
    parser.add_argument("-a", "--agent", help = "agent name")
    parser.add_argument("-c", "--clear", help = "clear old data")
 
    # Read arguments from command line
    args = parser.parse_args()

    agent_name = args.agent
    clear      = args.clear

    base_dir = Path("../agent_code/{}".format(agent_name))

    if not agent_name:
        print("AGENT NAME REQUIRED")
        return
    if not base_dir.exists():
        print("AGENT DOES NOT EXIST")
        return

   
    img_path = Path(os.path.join(base_dir, "logs/analysis/imgs"))
    img_path.mkdir(parents=True, exist_ok=True)    

    # clear data
    if clear == "true":
        shutil.rmtree(os.path.abspath(img_path))
        #shutil.move(os.path.abspath(img_path), os.path.abspath(os.path.join(base_dir, "logs/analysis/imgs_old")))

    img_path.mkdir(parents=True, exist_ok=True)    


    history_path = os.path.join(base_dir, "logs/analysis/history.npy")


    # open files in read mode
    with open(history_path, "rb") as file:
        history = pickle.load(file)


    action_history   = {}
    lifetime_history = {}


    num_of_games = 10*len(history.keys())
    current_game_counter = 0
    last_round_counter = 0
    for i, key in enumerate(history):
        for game_counter, round_counter, actions, reward in history[key]:

            # über training sessions hinweg zählen
            game_counter = game_counter + i*10 - 1

            for action in actions:

                if not action in action_history:
                    action_history[action] = np.zeros(num_of_games)

                action_history[action][game_counter] += 1

            if round_counter == 1 and game_counter != 0:
                lifetime_history[game_counter] = last_round_counter

            last_round_counter = round_counter

        lifetime_history[game_counter] =  round_counter


    del action_history["KILLED_SELF"]
    del action_history["GOT_KILLED"]
    del action_history["BOMB_EXPLODED"]

    actions = action_history.keys()
    action_values = np.asmatrix(list(action_history.values()))

    # TODO: add wievielter platz agent geworden ist
    # TODO: add analysis of one single game

    ### PIECHARTS

    num_of_files = int(np.ceil(num_of_games/20))

    for file_num in range(num_of_files):

        fig, ax = plt.subplots(4, 5, figsize=(19, 9), subplot_kw=dict(aspect="equal"))
        fig.suptitle("Action intensity of {} games - {}".format(agent_name, file_num))

        for grid_num, i in enumerate(range(20*file_num, min(20*(file_num + 1), num_of_games))):

            values = np.squeeze(np.array(action_values[:, i]))
            wedges, texts, autotexts = ax[grid_num//5, grid_num%5].pie(values, labels=None, autopct='', shadow=False, startangle=90)

            ax[grid_num//5, grid_num%5].title.set_text("Game {}".format(i + 1))
        
        ax[0, 4].legend(wedges, actions, title="Actions", loc="upper right", bbox_to_anchor=(2.8, 1.7))

        
        plt.savefig(os.path.join(img_path, "pycharts-all-actions_{}.png".format(file_num)))


    ### ACTION TIMELINE PLOTS

    colors = plt.rcParams["axes.prop_cycle"]()

    num_of_actions = len(actions)

    fig, ax = plt.subplots(int(np.ceil(num_of_actions/4)), 4, figsize=(19, 9), sharex=True)
    fig.suptitle("Action distributions of {} over all games".format(agent_name))
    plt.tight_layout()

    for i, action in enumerate(actions):
        values = action_history[action]
        c = next(colors)["color"]

        ax[i//4, i%4].fill_between(np.arange(len(values)), values, step="pre", alpha=0.4, color=c)
        ax[i//4, i%4].step(np.arange(len(values)), values, color=c)
        ax[i//4, i%4].title.set_text(action)

    fig.supxlabel('#game')
    fig.supylabel('#actions')
    plt.savefig(os.path.join(img_path, "distribution-all-actions.png"))


    ### LIFETIME PLOT
    fig = plt.figure(figsize=(19, 9))

    rounds = list(lifetime_history.keys())
    lifetimes = list(lifetime_history.values())

    plt.fill_between(np.arange(len(lifetimes)), lifetimes, step="pre", alpha=0.4, color="navy")
    plt.step(np.arange(len(lifetimes)), lifetimes, c="navy")

    plt.xlabel("#game")
    plt.ylabel("lifetime [#rounds]")

    plt.title("Agent {}: Lifetime".format(agent_name))

    plt.savefig(os.path.join(img_path, "lifetime.png"))


if __name__ == "__main__":
    main()