from nlrl.envs.frozen_lake.gym_frozenlake.env import FrozenLakeEnv, FrozenLakeEnvConfig

class FrozenLakeEnv_Wrapper(FrozenLakeEnv):
    def __init__(self, env_config, **kwargs):
        frozenlake_config = FrozenLakeEnvConfig()
        frozenlake_config.is_slippery = env_config.is_slippery
        super(FrozenLakeEnv_Wrapper, self).__init__(frozenlake_config)

    def set_state(self, state=None, player=None):



        raise NotImplementedError

    def get_available_action(self):
        return
