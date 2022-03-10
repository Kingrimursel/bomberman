#!/bin/python3

import argparse
import pickle

from pathlib import Path
import numpy as np



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
    if not base_dir.is_dir():
        print("AGENT DOES NOT EXIST")
        return


    # create analysis directory
    Path("../agent_code/{}/logs/analysis".format(agent_name)).mkdir(exist_ok=True)    


    round_counter = 0
    game_counter  = 0

    datetime = None

    history_path = Path("../agent_code/{}/logs/analysis/history.npy".format(agent_name))
    placement_path = Path("../agent_code/{}/logs/analysis/placement.npy".format(agent_name))

    if not history_path.is_file() or clear == "true":
        with open(history_path, "wb") as file:
            pickle.dump({}, file)

    if not placement_path.is_file() or clear == "true":
        with open(placement_path, "wb") as file:
            pickle.dump({}, file)

    # open files in read mode
    with open(history_path, "rb") as file:
        history = pickle.load(file)

    # open files in read mode
    with open(placement_path, "rb") as file:
        placement = pickle.load(file)

    with open("../agent_code/{}/logs/{}.log".format(agent_name, agent_name), "r") as file:
        last_round_counter = 0
        for line_nr, line in enumerate(file):
            line_elements = line.split(" ")

            if len(line_elements) <= 1:
                continue

            date = line_elements[0]
            time = line_elements[1]
            level = line_elements[3][:-1]

            if line_nr == 0:
                datetime = "{}/{}".format(date, time.split(",")[0])

                if not datetime in history.keys():
                    history[datetime]   = []
                    placement[datetime] = []

                else:
                    print("FILE ALREADY READ IN")
                    return

            message = line_elements[4:]
            # remove \n
            message[-1] = message[-1][:-1] 


            # update game counter
            if " ".join(message[:3]) == "Encountered game event(s)":
                round_counter = message[-1]
                if round_counter == "1":
                    game_counter += 1

            agents_placement = None
            if " ".join(message[:2]) == "Agents placement:":
                agents_placement = int(message[-1])

            if message[0] == "Awarded":
                reward = message[1]

                events = message[4:]

                for i, event in enumerate(events):
                    event = event.replace(",", "")
                    events[i] = event

                history[datetime].append((int(game_counter), int(round_counter), events, int(reward)))

            last_round_counter = round_counter

            if agents_placement:
                placement[datetime].append(agents_placement)


    with open(history_path, 'wb') as file:
        pickle.dump(history, file)

    with open(placement_path, 'wb') as file:
        pickle.dump(placement, file)

if __name__ == "__main__":
    main()