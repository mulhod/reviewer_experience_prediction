#!/bin/zsh
# Run incremental learning experiments for a set of games (evaluated on
# themselves, respectively) with each model trained on only NLP features. Use
# -h/--help option to see description and usage details.

set -eu

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
NON_NLP_FEATURES="none"
OUTPUT_DIR="$(pwd)/inc_learning_experiments"
SAMPLES_PER_ROUND="100"
TEST_LIMIT="1000"
ROUNDS="25"
PREDICTION_LABEL="total_game_hours_bin"
N_BINS="NULL"
BIN_FACTOR="NULL"

# Function for printing usage details
usage_details () {
    
    cat <<EOF
Usage: inc_learning_all.sh GAMES [OPTIONS]...

Run incremental learning experiments on all games.

positional arguments:

 GAMES                          comma-separated list of games: $(echo ${GAME_FILES} | sed 's: :, :g')

optional arguments:

 --out_dir=PATH                 PATH to directory to output incremental learning reports (default: ${OUTPUT_DIR})
 --samples_per_round=N_SAMPLES  N_SAMPLES, number of training samples to use per round (default: 100)
 --test_limit=N_SAMPLES         N_SAMPLES, number of samples to use for evaluation (default: 1000)
 --prediction_label=LABEL       LABEL to use for prediction label (defaults to "total_game_hours_bin"): $(echo ${LABELS} | sed 's: :, :g')
 --non_nlp_features=ALL_OR_NONE use "all" to use all non-NLP features or "none" to use none of them (default: "none")
 --nbins=NUMBER                 NUMBER of bins to divide the distribution of values corresponding to the prediction label into
 --bin_factor=FACTOR            floating point FACTOR (> 0) by which the sizes of the bins (set above) decrease or increase in terms of their range as the prediction label value increases (to set all bins to the same size, the default behavior, set this to 1.0)
 --help/-h                      print help
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
        OUTPUT_DIR=$(echo $1 | awk -F= '{print $2}')
        ;;
    --samples_per_round=*)
        SAMPLES_PER_ROUND=$(echo $1 | awk -F= '{print $2}')
        [[ $(echo ${SAMPLES_PER_ROUND} | grep -P "^[1-9][0-9]*$" | wc -l) -ne 1 ]] && {
            echo "ERROR: ${SAMPLES_PER_ROUND} not an integer. Exiting.\n"
            usage_details
            exit 1
        }
        ;;

    --test_limit=*)
        TEST_LIMIT=$(echo $1 | awk -F= '{print $2}')
        [[ $(echo ${TEST_LIMIT} | grep -P "^[1-9][0-9]*$" | wc -l) -ne 1 ]] && {
            echo "ERROR: ${TEST_LIMIT} not an integer. Exiting.\n"
            usage_details
            exit 1
        }
        ;;
    --prediction_label=*)
        PREDICTION_LABEL=`echo $1 | awk -F= '{print $2}'`
        for label in ${LABELS}; do
            
            if [[ $(echo ${LABELS} | grep -P "\b${label}\b" | wc -l) -ne 1 ]]; then
                
                echo "Unrecognized label: ${label}\n"
                echo "Exiting.\n"
                usage_details
                exit 1
                
            fi
            
        done
        ;;
    --non_nlp_features=*)
        NON_NLP_FEATURES=$(echo $1 | awk -F= '{print $2}')
        [[  ${NON_NLP_FEATURES} != "all" && ${NON_NLP_FEATURES} != "none" ]] && {
            echo "ERROR: --non_nlp_features must be set to either \"all\" " \
                 "or \"none\". You specified: ${NON_NLP_FEATURES}. Exiting.\n"
            usage_details
            exit 1
        }
        ;;
    --nbins=*)
        N_BINS=$(echo $1 | awk -F= '{print $2}')
        [[ $(echo ${N_BINS} | grep -P "^[1-9][0-9]*$" | wc -l) -ne 1 ]] && {
            echo "ERROR: ${N_BINS} not an integer. Exiting.\n"
            usage_details
            exit 1
        }
        ;;
    --bin_factor=*)
        BIN_FACTOR=$(echo $1 | awk -F= '{print $2}')
        [[ $(echo ${BIN_FACTOR} | grep -P "^[0-9]+\.[0-9]+$" | wc -l) -ne 1 \
            && ${BIN_FACTOR} -gt 0 ]] && {
            echo "ERROR: ${BIN_FACTOR} not a positive, non-zero floating " \
                 "point number. Exiting.\n"
            usage_details
            exit 1
        }
        ;;
    *)
        echo "ERROR: Unrecognized option: $1\n"
        echo "Exiting.\n"
        usage_details
        exit 1
        ;;
    esac
    shift
    
done

# Activate conda environment
source activate reviews

# Make output directory
mkdir -p ${OUTPUT_DIR}

for game in ${GAMES}; do
    
    echo "Conducting experiments with ${game}...\n"
    
    NON_NLP_FEATURES_STRING="non_nlp_features"
    if [[ ${NON_NLP_FEATURES} == "all" ]]; then
        
        NON_NLP_FEATURES_STRING="all_${NON_NLP_FEATURES_STRING}"
        
    else
        
        NON_NLP_FEATURES_STRING="no_${NON_NLP_FEATURES_STRING}"
        
    fi
    LOG="${OUTPUT_DIR}/${game}_${N_BINS}_bins_${BIN_FACTOR}_factor_${NON_NLP_FEATURES_STRING}.txt"
    CMD="learn --games ${game} --non_nlp_features ${NON_NLP_FEATURES} --output_dir ${OUTPUT_DIR} --rounds ${ROUNDS} --samples_per_round ${SAMPLES_PER_ROUND} --test_limit ${TEST_LIMIT} --prediction_label ${PREDICTION_LABEL} --nbins ${N_BINS} --bin_factor ${BIN_FACTOR} 2>! ${LOG}"
    # Get rid of --nbins/--bin_factor arguments if unspecified
    if [[ ${N_BINS} == "NULL" ]]; then
        
        CMD=$(echo ${CMD} | sed 's: --nbins::' | sed 's: NULL::')
        
    fi
    if [[ ${BIN_FACTOR} == "NULL" ]]; then
        
        CMD=$(echo ${CMD} | sed 's: --bin_factor::' | sed 's: NULL::')
        
    fi
    echo "${CMD}"
    eval "${CMD}"
    echo "Finished experiments with ${game}.\n"
    
done

echo "Complete.\n"
