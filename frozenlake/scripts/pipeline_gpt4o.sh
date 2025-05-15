#!/bin/bash
set -ex
export PYTHONPATH="$PWD:$PYTHONPATH"

TOP_K_SAMPLE=10
NUM_POLICY_SAMPLE=10
N_MC_TRAJ=4
NUM_ROLLOUTS=64
OPPONENT_POLICY_NAME="None"
NUM_TRAIN_EPOCH=64
N_HISTORY=3
EXP_BASE_DIR=./results-frozenlake-gpt4o/${2:-"exp"}
exp_dir="${EXP_BASE_DIR}/oppo_${OPPONENT_POLICY_NAME}_train_${NUM_TRAIN_EPOCH}_hist_${N_HISTORY}_sample_${NUM_POLICY_SAMPLE}_top_${TOP_K_SAMPLE}_nmc_${N_MC_TRAJ}_rollout_${NUM_ROLLOUTS}"
BASE_EVAL_TRAJ_PATH="${exp_dir}/data/eval/replay_buffer"
ENV_NAME="FrozenLakeEnv"


EVAL_TRAJ_PATH="${BASE_EVAL_TRAJ_PATH}_0.jsonl"
python3 frozenlake/eval.py \
    --env_name $ENV_NAME \
    --policy_name "LLM" \
    --opponent_policy_name $OPPONENT_POLICY_NAME \
    --replay_buffer_path $EVAL_TRAJ_PATH \
    --rollout_method scratch \
    --num_rollouts $NUM_ROLLOUTS \
    --env_parallel_num 32 \
    --model_path "gpt-4o" \
    --epsilon_greedy 0 \
    --temp 0

python3 nlrl/evaluate.py --data_dir ${exp_dir}/data/eval
