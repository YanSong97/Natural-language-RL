set -ex

# Experiment parameters
EXP=$(date +"%Y%m%d")
MODEL_PATH=${MODEL_PATH:-"path/to/llama-3.1-8b"}

bash ./frozenlake/scripts/pipeline_llama3.1_8b_prompting.sh \
	$EXP \
	$MODEL_PATH
