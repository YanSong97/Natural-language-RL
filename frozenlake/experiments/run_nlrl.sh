set -ex

# Experiment parameters
TOP_K_SAMPLE=10
NUM_POLICY_SAMPLE=10
N_MC_TRAJ=4
NUM_ROLLOUTS=512
OPPONENT_POLICY_NAME="None"
NUM_TRAIN_EPOCH=2
NUM_HISTORY=3
EXP=$(date +"%Y%m%d_llama_all_seed_slippery")
START_ITERATION_NUM=1
FINAL_ITERATION_NUM=31
ENV_NAME="FrozenLakeEnv"    # TicTacToeEnv

# Run pipeline
bash ./frozenlake/scripts/pipeline_nlac.sh \
    $TOP_K_SAMPLE \
	$NUM_POLICY_SAMPLE \
    $N_MC_TRAJ \
    $NUM_ROLLOUTS \
    $OPPONENT_POLICY_NAME \
    $NUM_TRAIN_EPOCH \
    $NUM_HISTORY \
    $EXP \
	$START_ITERATION_NUM \
	$FINAL_ITERATION_NUM \
	$ENV_NAME
