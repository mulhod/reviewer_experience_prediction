#!/bin/zsh
# Run incremental learning experiments for a set of games (evaluated on
# themselves, respectively) with each model trained on only NLP features. Use
# -h/--help option to see description and usage details.

set -e

# Variables and default values
DATA_DIR="$(dirname $(dirname $(readlink -f $0)))/data"
GAME_FILES=( $(ls ${DATA_DIR}/*jsonlines \
               | grep -v "sample" \
               | awk -F/ '{print $NF}' \
               | awk -F. '{print $1}' ) )
LABELS=( "num_guides" "num_games_owned" "num_friends" "num_voted_helpfulness" \
         "num_groups" "num_workshop_items" "num_reviews" "num_found_funny" \
         "friend_player_level" "num_badges" "num_found_helpful" \
         "num_screenshots" "num_found_unhelpful" "found_helpful_percentage" \
         "num_comments" "total_game_hours" "total_game_hours_bin" \
         "total_game_hours_last_two_weeks" "num_achievements_percentage" \
         "num_achievements_possible" )
OUTPUTDIR="$(pwd)/inc_learning_experiments"
SAMPLES_PER_ROUND="100"
TEST_LIMIT="1000"
ROUNDS="25"
PREDICTION_LABEL="total_game_hours_bin"

# Function for printing usage details
usage_details () {
    
    cat <<EOF
Usage: inc_learning_all_games.sh GAMES [OPTIONS]...

Run incremental learning experiments on all games.

positional arguments:

 GAMES                     comma-separated list of games: $(echo ${GAME_FILES} | sed 's: :, :g')

optional arguments:

 --out_dir=PATH            PATH to directory to output incremental learning reports (default: ${OUTPUTDIR})
 --samples_per_round       N_SAMPLES, number of training samples to use per round (default: 100)
 --test_limit              N_SAMPLES, number of samples to use for evaluation (default: 1000)
 --prediction_label=LABEL  LABEL to use for prediction label (defaults to "total_game_hours_bin"): $(echo ${LABELS} | sed 's: :, :g')
 --help/-h                 print help
EOF
    
}

# Get list of games to run experiments for
GAMES=( $(echo "$1" | sed 's:,: :g') )
shift

# Make sure games are valid games
for game in ${GAMES}; do
    
    if [[ $(echo ${GAME_FILES} | grep -P "\b${game}\b" | wc -l) -ne 1 ]]; then
        
        echo "Unrecognized game: ${game}\n"
        usage_details
        exit 1
        
    fi
    
done

# Get command-line arguments
while [ "$1" != "" ]; do
    
    case "$1" in
    -h|--help)
        usage_details
        exit 0
        ;;
    --out_dir=*)
        OUTPUTDIR=$(echo $1 | awk -F= '{print $2}')
        ;;
    --samples_per_round=*)
        SAMPLES_PER_ROUND=$(echo $1 | awk -F= '{print $2}')
        [[ $(echo ${SAMPLES_PER_ROUND} | grep -P "^[1-9][0-9]*$" | wc -l) -ne 1 ]] && {
            echo "ERROR: ${SAMPLES_PER_ROUND} not an integer. Exiting."
            usage_details
            exit 1
        }
        ;;

    --test_limit=*)
        TEST_LIMIT=$(echo $1 | awk -F= '{print $2}')
        [[ $(echo ${TEST_LIMIT} | grep -P "^[1-9][0-9]*$" | wc -l) -ne 1 ]] && {
            echo "ERROR: ${TEST_LIMIT} not an integer. Exiting."
            usage_details
            exit 1
        }
        ;;
    --prediction_label=*)
        PREDICTION_LABEL=`echo $1 | awk -F= '{print $2}'`
        for label in ${LABELS}; do
            
            if [[ $(echo ${LABELS} | grep -P "\b${label}\b" | wc -l) -ne 1 ]]; then
                
                echo "Unrecognized label: ${label}\n"
                usage_details
                exit 1
                
            fi
            
        done
        ;;
    *)
        echo "ERROR: Unrecognized option: $1"
        echo "Exiting."
        usage_details
        exit 1
        ;;
    esac
    shift
    
done

# Activate conda environment
#source activate reviews

# Make output directory
mkdir -p ${OUTPUTDIR}

for game in ${GAMES}; do
    
    echo "Conducting experiments with ${game}...\n"
    
    LOG="${OUTPUTDIR}/log_inc_learning_${game}.txt"
    eval "learn --games ${game} --output_dir ${OUTPUTDIR} --rounds ${ROUNDS} --samples_per_round ${SAMPLES_PER_ROUND} --test_limit ${TEST_LIMIT} --prediction_label ${PREDICTION_LABEL}" >! ${LOG}
    
    echo "Finished experiments with ${game}.\n"
    
done

echo "Complete.\n"