# Natural Language Reinforcement Learning

Reinforcement Learning (RL) mathematically formulates decision-making with Markov Decision Process (MDP). With MDPs, researchers have achieved remarkable breakthroughs across various domains, including games, robotics, and language models. This paper seeks a new possibility, Natural Language Reinforcement Learning (NLRL), by extending traditional MDP to natural language-based representation space. Specifically, NLRL innovatively redefines RL principles, including task objectives, policy, value function, Bellman equation, and policy iteration, into their language counterparts. With recent advancements in Large Language Models (LLMs), NLRL can be practically implemented to achieve RL-like policy and value improvement by either pure prompting or gradient-based training. Experiments over Maze, Breakthrough, Tic-tac-toe and FrozenLake games demonstrate the effectiveness, efficiency, and interpretability of the NLRL framework among diverse use cases. 

## Installation
```
conda create -n nlrl python==3.10
conda activate nlrl

git clone <REPO>.git
cd Natural-language-RL
pip install -r requirements.txt

# Install gym-tictactoe
pip3 install -e tictactoe/gym-tictactoe/.

# Install OpenSpiel
cd open_spiel
pip3 install -e .

# We use a hacky implementation that reuse Huggingface checkpoint loading
# to reload model and optimizer at each training iteration
# which needs to slightly modify the original code of HuggingFace Trainer
# Comment line 1912 - 1914 in anaconda3/envs/nlrl/lib/python3.10/site-packages/transformers/trainer.py
# Specifically comment out these 3 lines:
state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
if state.train_batch_size is not None:
    self._train_batch_size = state.train_batch_size

# Add current dir to PYTHONPATH
export PYTHONPATH=./
```

## Usage
The `nlrl` directory contains shared libraries for our four experiments: `maze`, `breakthrough`, `tictactoe`, `frozenlake`. Please refer to the README in each respective folder for specific usage instructions.
