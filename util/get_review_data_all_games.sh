#!/bin/zsh

# Usage: ./get_review_data_all_games.sh[ WAIT_TIME]
# WAIT_TIME: number of seconds to tell the program to sleep in between
# webpage requests (default: 30)

if [[ $# -eq 1 ]]; then
    
    WAIT_TIME=30
    
else
    
    WAIT_TIME=$2
    
fi

DATA_DIR="$(dirname $(dirname $(readlink -f $0)))/data"
GAME_FILES=( $(ls ${DATA_DIR}/*jsonlines \
               | grep -v "sample" \
               | awk -F/ '{print $NF}' ) )
wd=`pwd`
cd "${DATA_DIR}/.."
for game in ${GAME_FILES}; do
    
    echo "Getting review data for ${game}..."
    python3.4 util/get_review_data.py \
        --games ${game} \
        --wait ${WAIT_TIME} \
        -log logs/replog_get_review_data_${game}.txt &
    sleep 5s
    
done

cd "${wd}"
echo "Complete.\n"
