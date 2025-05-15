from nlrl.config import EnvConfig, PolicyConfig, LLMSamplingParams
from nlrl.utils import rollout, read_jsonl, write_jsonl
from nlrl.envs import get_env
from nlrl.policy import get_policy
from nlrl.policy.llm_ppo_policy import PPOLLMAgent
from tqdm import tqdm
import jsonlines
import argparse


# Deprecated
def collect_rollout_data(args):
    env = get_env(args.env_config)
    policy = get_policy(args.policy_config)
    boards = read_jsonl(args.data_path)

    replay_buffer = {}
    for ind, state in enumerate(boards):
        traj_data_list = []
        for i in range(args.value_data_config.rollout_steps):
            traj_data = rollout(env, policy, state=state["board"])
            traj_data_list.append(traj_data)
    replay_buffer[ind] = traj_data_list
    with jsonlines.open(args.replay_buffer_path, mode="w") as writer:
        for prompt in replay_buffer:
            writer.write(prompt)


def collect_batch_rollout_single_player(
    env_config, policy_config, opponent_policy_config, replay_buffer_path, num_rollout
):
    """
    Collect rollout data for a batch of boards.
    Start from scratch.
    The first player will be randomized.
    """
    batch_sample_size = env_config.batch_sample_size
    env_list = [get_env(env_config) for _ in range(batch_sample_size)]

    if "policy" in policy_config.model_path.lower() and policy_config.policy_name == "LLM_PPO":
        batch_policy = PPOLLMAgent.from_pretrained(
            pretrained_path=policy_config.model_path,
            env_config=EnvConfig(env_name=env_config.env_name),
            epsilon_greedy=policy_config.epsilon_greedy,
            temperature=policy_config.llm_config.temperature,
        )
    else:
        batch_policy = get_policy(policy_config)
    replay_buffer = []

    for batch_idx in tqdm(range(num_rollout // batch_sample_size), desc="Collecting"):
        traj_data = [
            {"state": [], "action": [], "reward": [], "turn": []}
            for _ in range(batch_sample_size)
        ]
        states = [env.reset() for env in env_list]
        dones = [False] * batch_sample_size
        for idx, state in enumerate(states):
            traj_data[idx]["state"].append(state)

        while not all(dones):
            current_state = []
            for i, (env, done) in enumerate(zip(env_list, dones)):
                if not done:
                    current_state.append(states[i])

            if not current_state:
                break

            action = batch_policy(current_state, available_actions_list=[[0,1,2,3] for state in current_state])      # default available action for FrozenLake
            # import pdb
            # pdb.set_trace()


            # print(f"action = {action}")
            # time.sleep(100)
            action_id = 0
            for i, (env, done) in enumerate(zip(env_list, dones)):
                if done:
                    continue
                next_state, reward, done, info = env.step(action[action_id]+1)      # input action start from index 1
                traj_data[i]["state"].append(next_state)
                traj_data[i]["action"].append(action[action_id])        # taking record of action from index 0
                traj_data[i]["reward"].append(reward)
                states[i] = next_state
                action_id += 1
                if done:
                    dones[i] = True
                    print(f"Env {i} is done.")

        replay_buffer += traj_data
    write_jsonl(replay_buffer, replay_buffer_path)
    print("Data collected successfully.")
    print("Data saved at:", replay_buffer_path)
    return replay_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # normal args
    parser.add_argument(
        "--replay_buffer_path", type=str, default="pipeline/data/replay_buffer.jsonl"
    )
    parser.add_argument(
        "--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--policy_name", type=str, default="LLM")
    parser.add_argument("--opponent_policy_name", type=str, default="Random")
    parser.add_argument("--env_name", type=str, default="TicTacToeEnv")
    parser.add_argument("--env_parallel_num", type=int, default=64)
    parser.add_argument(
        "--rollout_method", choices=["scratch", "initial"], default="scratch"
    )

    # args for collecting rollouts from given boards
    parser.add_argument(
        "--initial_board_path", type=str, default="pipeline/data/new_board.jsonl"
    )
    parser.add_argument("--rollout_per_state", type=int, default=1)
    # args for collecting rollouts from scratch
    parser.add_argument("--num_rollouts", type=int, default=512)

    # args for llm policy
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--epsilon_greedy", type=float, default=None)
    parser.add_argument("--prompt_logprobs", type=bool, default=False)
    parser.add_argument("--generate_mini_bz", type=int, default=None)

    parser.add_argument("--unslippery", action="store_true")
    parser.add_argument("--model_tp_size", type=int, default=1)

    args = parser.parse_args()

    if args.rollout_method == "scratch":
        assert args.num_rollouts % args.env_parallel_num == 0

    env_config = EnvConfig(
        env_name=args.env_name,
        batch_sample=args.env_parallel_num > 1,
        batch_sample_size=args.env_parallel_num,
        is_slippery=not args.unslippery
    )

    SamplingParams = LLMSamplingParams(
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        prompt_logprobs=args.prompt_logprobs,
    )
    policy_config = PolicyConfig(
        policy_name=args.policy_name,
        model_path=args.model_path,
        env_config=env_config,
        llm_config=SamplingParams,
        epsilon_greedy=args.epsilon_greedy,
        vllm_generate_mini_bz=args.generate_mini_bz,
        model_tp_size=args.model_tp_size
    )
    if args.opponent_policy_name.strip().lower() == "none":
        # single player environment
        opponent_policy_config = None
    else:
        opponent_policy_config = PolicyConfig(
            policy_name=args.opponent_policy_name, model_path=None, env_config=env_config
        )

    assert opponent_policy_config is None

    if env_config.batch_sample:
        if opponent_policy_config is None:
            collect_batch_rollout_single_player(
                env_config,
                policy_config,
                opponent_policy_config,
                args.replay_buffer_path,
                args.num_rollouts,
            )
        else:
            pass

    else:
        print("You are not using batch sampling.")
        raise NotImplementedError
        # collect_rollout_data(args)
