
from copy import deepcopy

import numpy as np
import random


def check_success(board):
    if "√" in board:
        return True, "succeed"
    elif "X" in board:
        return False, "fall"
    else:
        return False, "truncation"



#=============== Evaluation =========================
# For GPT3.5 and 4 sampling evaluation as baselines
FROZEN_LAKE_STATE_EVAL_PROMPT_GPT = """\
You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid.
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

Your task is to evaluate a given board position.
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

You should output your answer in the json format. Your answer consists of two elements:
- "thought": Let's think step by step. Generate your detailed thought process and evaluation of the agent's position and the distance towards the goal.
- "final_evaluation": Concisely judge how good the agent's current position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.


Here is the board position:
Board:
{state}
"""

FROZEN_LAKE_STATE_EVAL_PROMPT_GPT_SA = """\
You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid.
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

Your task is to evaluate a given board position and next action.
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

You should output your answer in the json format. Your answer consists of two elements:
- "thought": Let's think step by step. Generate your detailed thought process and evaluation of the agent's position after taking the possible action and the distance towards the goal.
- "final_evaluation": Judge how good the agent's position is after taking the action, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position after taking this action. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.


Here is the board position and next move:
Board:
{state}
Action:
The next move is {action}.
"""

#================= Decision ====================

FROZEN_LAKE_STATE_POLICY_SYSTEM_PROMPT_GPT = """\
You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid.
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

Your task is to choose the best move given board position.
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed thought process and plan for the next move.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 4, indicating the moving direction.
"""

QWEN_FROZEN_LAKE_STATE_POLICY_USER_PROMPT = """\
Here is the board position:
{state}. The available moves are {available_move}.

Write your answer in a json format.
"""

FROZEN_LAKE_STATE_POLICY_USER_PROMPT = """\
Here is the board position:
{state}. The available moves are {available_move}.
"""

#==================== In-context Example ===============================
FROZEN_LAKE_POLICY_EXAMPLE_USER_PROMPT = """\
Here is the board position:
__P_
_O__
___O
GO__
The available move positions are 1,2,3,4.
"""

FROZEN_LAKE_POLICY_EXAMPLE_ASSISTANT_PROMPT = """\
{"thought": "It appears that the initial board position
__P_
_O__
___O
GO__
demonstrate that the target position is on the left bottom direction and P can move to target by moving left first and then go down. Therefore, the best next move for P is to play action 1 to move left.", "best_move": 1}
"""

FROZEN_LAKE_POLICY_EXAMPLE_PROMPT = FROZEN_LAKE_POLICY_EXAMPLE_USER_PROMPT + "\n" + FROZEN_LAKE_POLICY_EXAMPLE_ASSISTANT_PROMPT


#========================== Monte-Carlo Estimate ====================================
MC_PROMPT = """You are a reinforcement learning agent of the game of FrozenLake. 
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

You are learning how to evaluate a board in the FrozenLake by playing the game and reflect the playing history. 
The following is a rollout history depicting a game in progress with a final result. 
{trajectory}
You do not know the value of the first board as you are a learner. 
You have to evaluate the board in hindsight based on the rollout sequence.
You have to review the board by reflecting on the sequence and the ultimate outcome. 
Do not just evaluate the state based on your knowledge because you are a learner and your knowledge is inaccurate. 
Just review the board, do not guess the potential move. 
Just review the board and do not give any advice unrelated to the board. Explicitly state your evaluation based on the rollout sequence. Point out the threatens and opportunities of each board based on the rollout sequence. """

MC_SYSTEM_PROMPT = """You are a player of the game of FrozenLake. 
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

You are learning how to evaluate a board in the FrozenLake by playing the game and reflect the playing history. 
The following is a rollout history depicting a game in progress with a final result.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed evaluation by analyzing the game from different perspectives. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies.
- "final_evaluation": After all your thought, finally judge how good the agent's current position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position after taking this action. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5."""


# scalar evaluation?
MC_EXAMPLE_USER_PROMPT = """The board to evaluate:
_P__
_O__
___O
GO__

For your reference, below is the rollout sequence:
After move 2 (Down), the board position is:
____
_X__
___O
GO__
The game is over. Agent fails due to falling into the hole."""

MC_EXAMPLE_ASSISTENT_PROMPT="""
{"thought": {"Reflection": "It appears that the initial board position
_P__
_O__
___O
GO__
was not favorable for P, as P is still 4 blocks away from the goal position G.", "Win probability": "The win/success probability for P is small.", "Threat": "There are three holes on the board. There is a hole right below P which can be dangerous if P choose to move down.", "Potential strategies": "Potential strategies for P can be 1 (left) followed by 2 (Down), 2 (Down), 2 (Down) to move to the goal position; or 3 (Right) followed by 2 (Down), 2 (Down), 1 (Left), 1 (Left), 2 (Down)."}
"final_evaluation": -0.4}
"""


MC_SYSTEM_PROMPT_SA = """You are a player of the game of FrozenLake. 
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

You are learning how to evaluate a (board, action) pair in the FrozenLake by playing the game given the (board, action) pair and reflect the playing history.
The playing history depicts a game in progress with a final result. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. Your should only rely on the playing history as context and don't evaluate game with your own judgement. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies.
- "final_evaluation": After all your thought, finally judge how good the agent's current position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position after taking this action. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5."""


MC_EXAMPLE_USER_PROMPT_SA="""The （board, action) to evaluate:
Board:
P___
_O__
___O
GO__
Action:
2 (Down).

For your reference, below is the rollout sequence 1 after this (board, action):
After move 2 (Down), the board position is:
____
PO__
___O
GO__

After move 3 (Right), the board position is:
____
_X__
___O
GO__
The game is over. Player fall into the hole and therefore fails.
"""

MC_EXAMPLE_ASSISTENT_PROMPT_SA="""
{"thought": {"Reflection": "It appears that the initial board position
P___
_O__
___O
GO__
and move 2 (Down)
was favorable for P, as P is one block closer to the goal position G.", "Win probability": "The win/success probability for P is larger.", "Threat": "There are three holes on the board. There is a hole only one block away from P after action has been taken, which can be dangerous if O choose to move right.", "Potential strategies": "Potential strategies for P include moving 2 (Down) twice to arrive at goal position."}
"final_evaluation": -0.3}
"""

STATE_TO_VALUE_PROMPT_SA = """The board to evaluate is:
Board:
{state}
Action:
The next move is {action}.
"""

NEW_TRAJ_BEGIN_PROMPT = (
    "\nFor your reference, below is the rollout sequence {idx} after this (board, action):"
)

CONCAT_PROMPT = (
    "\nAfter taking action {action}, the board position is \n{board}.\n"
)

STATE_TO_VALUE_PROMPT = """The board to evaluate is:
{state}.

For your reference, below is the rollout sequence:"""

STATE_TO_ACTION_PROMPT = (
    "The current board is: \n{state}.\n\nFor your reference, below is the rollout sequence:"
)

TD_PROMPT = """
You are a reinforcement learning agent of the game of FrozenLake. 
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.
You are learning how to evaluate a board in the FrozenLake by playing the game and reflecting the playing history. 
The following is a board and a next board. 

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

{trajectory}
"""

#=================== Policy Improvement ================================

POLICY_IMPROVEMENT_PROMPT = """\
You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid.
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

Your task is to evaluate a given board position.
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

Your task is to determine the best next move based on the given board position.
The evaluations of boards after possible moves are given.
DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move.
The evaluation for the next board is in the format of a json format, consisting of two elements:
- "thought": evaluation of the board position.
- "final_evaluation": Judge how good the agent's current position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.
{example_prompt}
Here is the board position:
{state}. The possible moves are {available_positions}.
The following are the boards after each possible move:
{next_states}

Now, please give your evaluation and the best next move based on the given board position {state}.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 4, indicating the moving direction.
Don't output extra information except for the json format.
"""


POLICY_IMPROVEMENT_SYSTEM_PROMPT_SA = """\
You are an expert agent at playing the game FrozenLake on a 4*4 board. FrozenLake is a single-player game played on a grid.
The goal is to avoid the hole and move to the goal. You may move to the unintended direction due to the slippery ice.

Your task is to evaluate a given board position.
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, √: player on goal

The possible actions are:
1:Left, 2:Down, 3:Right, 4:Up

Your task is to determine the best next move based on the given board position.
The evaluations of (board, action) pairs after possible moves are given.
DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move.
The evaluation for the next board is in the format of a json format, consisting of two elements:
- "thought": evaluation of the the board and action pair.
- "final_evaluation": After all your thought, finally judge how good the agent's position is compared to all the possible positions in the board, in terms of reaching the goal. Measure the goodness in terms of negative distance to the goal position. For example, 4 blocks away will gives -0.4; 1 blocks away will gives -0.1; if the agent fall into the hole, gives -5; if the agent has reach the goal, gives 5.
"""

POLICY_IMPROVEMENT_USER_PROMPT_SA = """\
Here is the board position:
{state}. The possible moves are {available_positions}.
The following are the boards after each possible move:
{next_states}

Now, please give your evaluation and the best next move based on the given board position {state}.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 4, indicating the moving direction.
Don't output extra information except for the json format.
"""




class EVAL_prompt:
    def __init__(self, prompt=FROZEN_LAKE_STATE_EVAL_PROMPT_GPT):
        self.prompt = prompt
        split = "Here is the board position:"
        self.system_prompt = FROZEN_LAKE_STATE_EVAL_PROMPT_GPT.split(split)[0]
        self.user_prompt = split + FROZEN_LAKE_STATE_EVAL_PROMPT_GPT.split(split)[1]

    def format_input(self, state, response_type="LLM"):
        board = state #state_to_board(state)
        if response_type == "LLM":
            prompts = [{"role": "system", "content": self.system_prompt}]
            prompts.append(
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        state=board
                    ),
                }
            )
            return {"state": state, "prompt": prompts}
        else:
            raise NotImplementedError
            return

    def __call__(self, state, response_type="LLM"):
        return self.format_input(state, response_type=response_type)


class EVAL_prompt_sa:
    def __init__(self):
        split = "Here is the board position and next move:"
        self.system_prompt = FROZEN_LAKE_STATE_EVAL_PROMPT_GPT_SA.split(split)[0]
        self.user_prompt = split + FROZEN_LAKE_STATE_EVAL_PROMPT_GPT_SA.split(split)[1]

    def format_input(self, state_action_dict, response_type="LLM"):
        state = state_action_dict["state"]
        action = state_action_dict["action"]
        board = state   #state_to_board(state)
        if response_type == "LLM":
            prompts = [{"role": "system", "content": self.system_prompt}]
            prompts.append(
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        state=board, action=action + 1
                    ),
                }
            )
            return {"state": state, "prompt": prompts}
        else:
            raise NotImplementedError
            return

    def __call__(self, state_action_dict, response_type="LLM"):
        return self.format_input(state_action_dict, response_type=response_type)


class POLICY_prompt(EVAL_prompt):
    def __init__(self, prompt=FROZEN_LAKE_STATE_POLICY_SYSTEM_PROMPT_GPT, template='llama'):
        super().__init__(prompt)

        self.template = template

    def format_input(
        self, state, example_prompt, available_actions=None, response_type="LLM"
    ):

        board = state   #state_to_board(state)
        if available_actions is None:
            available_actions = [1,2,3,4]   #[i + 1 for i in range(9) if state[0][i] == 0]
        else:
            available_actions = [i + 1 for i in available_actions]      # the provided avail action is from index 0



        if response_type == "LLM":
            prompts = [
                {
                    "role": "system",
                    "content": FROZEN_LAKE_STATE_POLICY_SYSTEM_PROMPT_GPT
                    + "### EXAMPLE:\nuser:"
                    + FROZEN_LAKE_POLICY_EXAMPLE_USER_PROMPT
                    + "assistant:"
                    + FROZEN_LAKE_POLICY_EXAMPLE_ASSISTANT_PROMPT,
                }
            ]

            if self.template=='llama':
                _policy_user_prompt = FROZEN_LAKE_STATE_POLICY_USER_PROMPT
            elif self.template=='qwen':
                _policy_user_prompt = QWEN_FROZEN_LAKE_STATE_POLICY_USER_PROMPT
            else:
                raise NotImplementedError

            prompts.append(
                {
                    "role": "user",
                    "content": _policy_user_prompt.format(
                        state=board,
                        available_move=available_actions,
                    ),
                }
            )
            return {"state": state, "prompt": prompts}
        elif response_type == "llama":
            # example = "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>" + TIC_TAC_TOE_POLICY_EXAMPLE_USER_PROMPT + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>" + TIC_TAC_TOE_POLICY_EXAMPLE_ASSISTANT_PROMPT
            example = ""
            prompts = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                + FROZEN_LAKE_STATE_POLICY_SYSTEM_PROMPT_GPT
                + example
                + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
                + FROZEN_LAKE_STATE_POLICY_USER_PROMPT.format(
                    state=board,
                    available_move=available_actions,
                )
                + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
            )

            return prompts

    def __call__(
        self,
        state,
        example_prompt=FROZEN_LAKE_POLICY_EXAMPLE_PROMPT,
        available_actions=None,
        response_type="LLM",
    ):
        return self.format_input(
            state,
            example_prompt=example_prompt,
            available_actions=available_actions,
            response_type=response_type,
        )





class MC_prompt:
    def __init__(self, prompt=MC_PROMPT):
        self.prompt = prompt

    def format_input(self, traj_data):
        message = [
            {
                "role": "system",
                "content": MC_SYSTEM_PROMPT
                + "### EXAMPLE:\nuser:"
                + MC_EXAMPLE_USER_PROMPT
                + "assistant:"
                + MC_EXAMPLE_ASSISTENT_PROMPT,
            }
        ]
        trajectory_prompt = self.get_trajectory_prompt(traj_data)
        message.append({"role": "user", "content": trajectory_prompt})
        return message

    def get_trajectory_prompt(self, traj_data):
        trajectory_prompt = ""
        states = traj_data["state"]
        actions = traj_data["action"]
        for ind in range(len(states)):
            state = states[ind]
            board = state  #state_to_board(state)
            player = "X" if state[1] == "O" else "O"
            if ind == 0:
                trajectory_prompt += STATE_TO_VALUE_PROMPT.format(
                    player=state[1], state=board
                )
            else:
                action = (
                    actions[ind - 1] + 1
                )  # +1 because action id + 1 = the digit shown on the board
                trajectory_prompt += CONCAT_PROMPT.format(
                    player=player, board=board, action=action
                )

        final_board = traj_data["state"][-1][0]  #convert_input_to_blocks(traj_data["state"][-1][0])
        result, end_reason = check_success(final_board)
        if result:
            trajectory_prompt += "The game is over. Player has reach the goal position and therefore succeeds.\n\n"
        else:
            if end_reason=="fall":
                trajectory_prompt += "The game is over. Player fall into the hole and therefore fails.\n\n"
            elif end_reason=='truncation':
                trajectory_prompt += "The game is over. Player has reach maximum number of move and therefore fails.\n\n"
            else:
                raise NotImplementedError(f"result={result}, reason={end_reason}")

        return trajectory_prompt

    def __call__(self, traj_data):
        return self.format_input(traj_data)


class MC_prompt_sa:
    def __init__(self, prompt=MC_PROMPT, add_example=True):
        self.prompt = prompt
        self.add_example = add_example

    def format_input(self, traj_data):
        if self.add_example:
            message = [
                {
                    "role": "system",
                    "content": MC_SYSTEM_PROMPT_SA
                    + "### EXAMPLE:\nuser:"
                    + MC_EXAMPLE_USER_PROMPT_SA
                    + "assistant:"
                    + MC_EXAMPLE_ASSISTENT_PROMPT_SA,
                }
            ]
        else:
            message = [{"role": "system", "content": MC_SYSTEM_PROMPT_SA}]
        trajectory_prompt = self.get_trajectory_prompt(traj_data)
        message.append({"role": "user", "content": trajectory_prompt})
        return message

    def get_trajectory_prompt(self, traj_list):
        trajectory_prompt = ""
        for i_traj, traj in enumerate(traj_list):
            states = traj["state"]
            actions = traj["action"]
            for ind in range(len(states)):
                state = states[ind]
                board = state  #state_to_board(state)
                player = "X" if state[1] == "O" else "O"
                if ind == 0:
                    if i_traj == 0:
                        trajectory_prompt += STATE_TO_VALUE_PROMPT_SA.format(
                            player=state[1], state=board, action=actions[0] + 1
                        )
                    trajectory_prompt += NEW_TRAJ_BEGIN_PROMPT.format(
                        idx=i_traj + 1
                    )  # to make it start from 1
                else:
                    action = (
                        actions[ind - 1] + 1
                    )  # +1 because action id + 1 = the digit shown on the board
                    trajectory_prompt += CONCAT_PROMPT.format(
                        player=player, board=board, action=action
                    )
            final_board = traj["state"][-1]  #convert_input_to_blocks(traj["state"][-1][0])

            result, end_reason = check_success(final_board)
            if result:
                trajectory_prompt += "The game is over. Player has reach the goal position and therefore succeeds.\n\n"
            else:
                if end_reason == "fall":
                    trajectory_prompt += "The game is over. Player fall into the hole and therefore fails.\n\n"
                elif end_reason == 'truncation':
                    trajectory_prompt += "The game is over. Player has reach maximum number of move and therefore fails.\n\n"
                else:
                    raise NotImplementedError(f"result={result}, reason={end_reason}")

        trajectory_prompt += (
            "Now generate your evaluation for the (board, action) pair."
        )
        return trajectory_prompt

    def __call__(self, traj_data):
        return self.format_input(traj_data)


class POLICY_IMPROVEMENT_prompt_sa:

    def __init__(self, prompt=POLICY_IMPROVEMENT_PROMPT):
        self.prompt = prompt
        self.system_prompt = POLICY_IMPROVEMENT_SYSTEM_PROMPT_SA
        self.user_prompt = POLICY_IMPROVEMENT_USER_PROMPT_SA

    def format_input_v2(
        self, state, example_prompt, actions, next_states, response_type="LLM"
    ):
        board = state #state_to_board(state)
        if actions is not None:
            available_positions = actions
        else:
            available_positions = [i + 1 for i in range(9) if state[0][i] == 0]
            assert len(available_positions) == len(next_states)
        current_player = state[-1]
        values = []
        for i, next_board in enumerate(next_states):
            values.append(
                f"\n### Evaluation for taking action {available_positions[i]}:\n"
                + next_board["value"]
            )
        if response_type == "LLM":
            prompt = [{"role": "system", "content": self.system_prompt}]
            prompt.append(
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        state=board,
                        next_player=current_player,
                        available_positions=available_positions,
                        next_states="\n".join(values),
                        example_prompt=example_prompt,
                    ),
                }
            )
            return {"state": state, "prompt": prompt}
        elif response_type == "llama":
            prompt = self.prompt.format(
                state=board,
                next_player=current_player,
                available_positions=available_positions,
                next_states="\n".join(values),
                example_prompt=example_prompt,
            )
            return prompt

    def get_state_action_pair(state):
        raise NotImplementedError
        state_action_pairs = []
        for i in range(9):
            if state[0][i] == 0:
                state_action_pairs.append({"state": state, "action": i})
        return state_action_pairs

    def __call__(
        self,
        state,
        example_prompt=None,
        actions=None,
        next_states=None,
        response_type="LLM",
    ):
        return self.format_input_v2(
            state, example_prompt, actions, next_states, response_type=response_type
        )



