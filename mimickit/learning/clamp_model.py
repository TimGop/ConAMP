import torch
import numpy as np
import learning.amp_model as amp_model
from learning.nets.residual_block import ResidualBlock
import util.torch_util as torch_util
import torch.nn as nn


def _apply_lecun_init(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # Equivalent to Flax variance_scaling(1/3, "fan_in", "uniform")
            fan_in = m.weight.size(1)
            bound = np.sqrt(1.0 / fan_in)  # 3 * (1/3) / fan_in
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CLAMPModel(amp_model.AMPModel):
    def __init__(self, config, env):
        # We need to know how many dimensions to slice off the observation
        self._task_dim = config.get("task_dim", 2)
        self._embed_dim = config.get("embed_dim", 64)
        self._global_observations = False
        super().__init__(config, env)
        self._activation = nn.ReLU if config.get("use_relu", True) else nn.SiLU
        return

    def activate_global_observations(self):
        self._global_observations = True

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

        phi_in_dim = (obs_dim - self._task_dim) + action_dim if not self._global_observations else obs_dim + action_dim
        psi_in_dim = self._task_dim

        # Grab architecture hyperparams, matching the Jax defaults
        network_width = config.get("infonce_hidden_dim", 1024)
        network_depth = config.get("infonce_depth", 4)
        embed_dim = config.get("embed_dim", 64)

        def build_encoder(in_dim):
            layers = []

            # 1. Initial layer (Dense -> Norm -> Act)
            layers.extend([
                nn.Linear(in_dim, network_width),
                nn.LayerNorm(network_width),
                self._activation()
            ])

            # 2. Residual blocks
            num_blocks = network_depth // 4
            for _ in range(num_blocks):
                # Pass self._activation so the block can instantiate fresh activation layers
                layers.append(ResidualBlock(network_width, self._activation))

            # 3. Final projection layer (No norm, no act)
            layers.append(nn.Linear(network_width, embed_dim))

            return nn.Sequential(*layers)

        self._phi_net = build_encoder(phi_in_dim)
        self._psi_net = build_encoder(psi_in_dim)

        # Optional: Emulate the exact Lecun Uniform initialization from Flax
        _apply_lecun_init(self._phi_net)
        _apply_lecun_init(self._psi_net)
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