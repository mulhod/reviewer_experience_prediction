#!/bin/zsh

# Usage: ./get_review_data_all_games.sh[ WAIT_TIME]
# WAIT_TIME: number of seconds to tell the program to sleep in between
# webpage requests (default: 30)

if [[ $# -eq 1 ]]; then
    
    WAIT_TIME=30
    
else
    
    WAIT_TIME=$2
    
fi

python3.4 util/get_review_data.py \
    --games Warframe \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Warframe.txt &
python3.4 util/get_review_data.py \
    --games The_Elder_Scrolls_V \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_The_Elder_Scrolls_V.txt &
python3.4 util/get_review_data.py \
    --games Team_Fortress_2 \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Team_Fortress_2.txt &
python3.4 util/get_review_data.py \
    --games Sid_Meiers_Civilization_5 \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Sid_Meiers_Civilization_5.txt &
python3.4 util/get_review_data.py \
    --games Grand_Theft_Auto_V \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Grand_Theft_Auto_V.txt &
python3.4 util/get_review_data.py \
    --games Garrys_Mod \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Garrys_Mod.txt &
python3.4 util/get_review_data.py \
    --games Football_Manager_2015 \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Football_Manager_2015.txt &
python3.4 util/get_review_data.py \
    --games Arma_3 \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Arma_3.txt &
python3.4 util/get_review_data.py \
    --games Dota_2 \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Dota_2.txt &
python3.4 util/get_review_data.py \
    --games Counter_Strike_Global_Offensive \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Counter_Strike_Global_Offensive.txt &
python3.4 util/get_review_data.py \
    --games Counter_Strike \
    --wait ${WAIT_TIME} \
    -log logs/replog_get_review_data_Counter_Strike.txt &