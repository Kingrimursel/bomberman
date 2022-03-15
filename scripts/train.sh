#!/bin/bash

# AUTOMATES MULTIPLE TRAINING SESSIONS

# INPUT:
#   -a: agent name (e.g. my_agent)
#   -i: number of iterations (e.g. 200)
#   -n: if it has value [new], then a new model is created
#   -s: scenario (coin-heaven or classic)

# BEISPIEL: ./train.sh -a own_coin -i 4 -n true -s coin-heaven

cd ..

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# read and safe command line arguments
while getopts ":a:i:n:s:" opt; do
  case $opt in
    a) agent="$OPTARG"
    ;;
    i) its="$OPTARG"
    ;;
    n) new="$OPTARG"
    ;;
    s) scenario="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


if [ "$scenario" != "coin-heaven" ];
then
  scenario=classic
fi

# if new is passed, create new model
if [ "$new" == true ];
then
  echo -e "${GREEN}creating new model...${NC}"
  mv agent_code/$agent/my-saved-model.pt agent_code/$agent/my-saved-model_old.pt
fi

cd scripts

# run iterations
for i in $(seq $its); do
  # train
  cd ..
	echo -e "${RED}training:  $i of $its ...${NC}"
	python3 main.py play --no-gui --my-agent $agent --train 1 --scenario $scenario

  cd scripts
  echo -e "${PURPLE}analysing logs...${NC}"
  if [ "$new" == true ] && [ ${i} == 1 ];
  then
    ./analyze_logs.py -a $agent -c true
  else
    ./analyze_logs.py -a $agent
  fi
done


echo -e "${BLUE}processing log data...${NC}"

if [ "$new" == true ];
then
  ./process_log_data.py -a $agent -c  true
else
  ./process_log_data.py -a $agent

fi

cd ..


# presenting
echo -e "${GREEN}presenting result...${NC}"
python3 main.py play --train 1 --my-agent $agent --scenario $scenario

