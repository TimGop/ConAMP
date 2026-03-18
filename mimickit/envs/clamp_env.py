import envs.task_location_env as task_location_env


class CLAMPEnv(task_location_env.TaskLocationEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def _update_reward(self):
        # Task reward is now learned!
        return

    def _update_info(self, env_ids=None):
        # Call the parent class to preserve tracking error logic
        super()._update_info(env_ids)

        # Fetch the absolute global coordinates for all environments
        char_id = self._get_char_id()

        # Returns tensors of shape [num_envs, ...]
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        tar_pos = self._tar_pos

        # Inject into the info dictionary
        self._info["root_pos"] = root_pos
        self._info["root_rot"] = root_rot
        self._info["desired_goal"] = tar_pos

        return