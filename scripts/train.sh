#!/bin/bash

# AUTOMATES MULTIPLE TRAINING SESSIONS

# INPUT:
#   -a: agent name (e.g. my_agent)
#   -i: number of iterations (e.g. 200)
#   -n: if it has value [new], then a new model is created

cd ..

RED='\033[0;31m'
NC='\033[0m'

# read and safe command line arguments
while getopts ":a:i:n:" opt; do
  case $opt in
    a) agent="$OPTARG"
    ;;
    i) its="$OPTARG"
    ;;
    n) new="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# if new is passed, create new model
if [ "$new" == true ];
then
  mv agent_code/$agent/my-saved-model.pt agent_code/$agent/my-saved-model_old.pt
fi


# run iterations
for i in $(seq $its); do
	echo -e "${RED} $i of $its ${NC}"
	python3 main.py play --no-gui --my-agent $agent --train 1
done

