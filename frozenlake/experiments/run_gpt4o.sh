set -ex

# Experiment parameters
EXP=$(date +"%Y%m%d")

bash ./frozenlake/scripts/pipeline_gpt4o.sh \
	$EXP
