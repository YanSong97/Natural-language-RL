set -ex

# Experiment parameters
OPPONENT_POLICY_NAME="Random"
EXP=$(date +"%Y%m%d")
MODEL_PATH=${MODEL_PATH:-"path/to/llama-3.1-70b"}

bash ./tictactoe/scripts/pipeline_llama3.1_70b_prompting.sh \
	$OPPONENT_POLICY_NAME \
	$EXP \
	$MODEL_PATH
