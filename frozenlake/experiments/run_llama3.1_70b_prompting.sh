set -ex

# Experiment parameters
EXP=$(date +"%Y%m%d")
MODEL_PATH=${MODEL_PATH:-"path/to/llama-3.1-70b"}

bash ./frozenlake/scripts/pipeline_llama3.1_70b_prompting.sh \
	$EXP \
	$MODEL_PATH
