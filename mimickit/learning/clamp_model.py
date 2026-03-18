import torch
import numpy as np
import learning.amp_model as amp_model
import util.torch_util as torch_util


class CLAMPModel(amp_model.AMPModel):
    def __init__(self, config, env):
        # We need to know how many dimensions to slice off the observation
        self._task_dim = config.get("task_dim", 2)
        self._embed_dim = config.get("embed_dim", 64)
        super().__init__(config, env)
        return

    def _build_nets(self, config, env):
        # Build Actor, Critic (from PPOModel) and Discriminator (from AMPModel)
        super()._build_nets(config, env)

        # Build our new InfoNCE networks
        self._build_infonce_nets(config, env)
        return

    def _build_infonce_nets(self, config, env):
        obs_space = env.get_obs_space()
        action_space = env.get_action_space()

        obs_dim = np.prod(obs_space.shape)
        action_dim = np.prod(action_space.shape)

        phi_in_dim = (obs_dim - self._task_dim) + action_dim
        psi_in_dim = self._task_dim

        # Increase this in your YAML!
        hidden_dim = config.get("infonce_hidden_dim", 1024)

        self._phi_net = torch.nn.Sequential(
            torch.nn.Linear(phi_in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            self._activation(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            self._activation(),
            torch.nn.Linear(hidden_dim, self._embed_dim)
        )

        self._psi_net = torch.nn.Sequential(
            torch.nn.Linear(psi_in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            self._activation(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            self._activation(),
            torch.nn.Linear(hidden_dim, self._embed_dim)
        )
        return

    def eval_phi(self, s_proprio, action):
        x = torch.cat([s_proprio, action], dim=-1)
        return self._phi_net(x)

    def eval_psi(self, goal):
        return self._psi_net(goal)

    def get_infonce_params(self):
        # Return parameters for the InfoNCE optimizer
        params = list(self._phi_net.parameters()) + list(self._psi_net.parameters())
        return params