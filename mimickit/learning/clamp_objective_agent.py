import os
import time
import torch
import torch.nn.functional as F

import learning.mp_optimizer as mp_optimizer
import util.mp_util as mp_util
import util.torch_util as torch_util
import learning.clamp_model as clamp_model
import learning.amp_agent as amp_agent
from learning.trajectory_buffer import TrajectoryBuffer
from envs.base_env import DoneFlags

class CLAMPObjectiveAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._task_dim = config.get("task_dim", 2)
        self._logsumexp_penalty_coeff = config.get("logsumexp_penalty_coeff", 0.1)

        # Curriculum scheduling parameters
        self._warmup_steps = config.get("warmup_steps", 1_000_000)
        self._anneal_steps = config.get("anneal_steps", 2_000_000)
        self._target_task_weight = config.get("target_task_weight", 0.5)
        self._target_style_weight = config.get("target_style_weight", 0.5)

        self._task_reward_weight = 0.0
        self._disc_reward_weight = 1.0
        self._curriculum_total_steps = 0

        # InfoNCE Training Config
        self._infonce_epochs = config.get("infonce_epochs", 5)
        self._infonce_batch_size = config.get("infonce_batch_size", 256)
        return

    def _update_reward_weights(self, num_steps_this_iteration):
        """Dynamically adjusts the task and style reward weights based on the step count."""
        self._curriculum_total_steps += num_steps_this_iteration
        step = self._curriculum_total_steps

        if step <= self._warmup_steps:
            self._task_reward_weight = 0.0
            self._disc_reward_weight = 1.0
        elif step >= (self._warmup_steps + self._anneal_steps):
            self._task_reward_weight = self._target_task_weight
            self._disc_reward_weight = self._target_style_weight
        else:
            alpha = (step - self._warmup_steps) / self._anneal_steps
            self._task_reward_weight = alpha * self._target_task_weight
            self._disc_reward_weight = 1.0 + alpha * (self._target_style_weight - 1.0)
        return

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)

        root_pos = info["root_pos"]
        root_rot = info["root_rot"]
        desired_goal = info["desired_goal"]

        self._exp_buffer.record("root_pos", root_pos)
        self._exp_buffer.record("root_rot", root_rot)
        self._exp_buffer.record("desired_goal", desired_goal)

        for env_idx in range(self.get_num_envs()):
            self._active_episodes[env_idx]["curr_obs"].append(obs[env_idx].clone().detach())
            self._active_episodes[env_idx]["curr_action"].append(action[env_idx].clone().detach())
            self._active_episodes[env_idx]["curr_root_pos"].append(root_pos[env_idx].clone().detach())
            self._active_episodes[env_idx]["curr_root_rot"].append(root_rot[env_idx].clone().detach())
        return

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)

        traj_buffer_size = config.get("traj_buffer_size", 1000)
        max_ep_len = config.get("max_ep_len", 1200)

        self._traj_buffer = TrajectoryBuffer(
            max_episodes=traj_buffer_size,
            max_ep_len=max_ep_len,
            device=self._device,
            gamma=config.get("discount", 0.99)
        )

        self._active_episodes = [
            {"curr_obs": [], "curr_action": [], "curr_root_pos": [], "curr_root_rot": []}
            for _ in range(self.get_num_envs())
        ]
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = clamp_model.CLAMPModel(model_config, self._env)
        self._model.activate_global_observations()
        return

    def _build_optimizer(self, config):
        super()._build_optimizer(config)

        infonce_config = config["infonce_optimizer"]
        infonce_params = list(self._model.get_infonce_params())
        infonce_params = [p for p in infonce_params if p.requires_grad]

        self._infonce_optimizer = mp_optimizer.MPOptimizer(infonce_config, infonce_params)
        return

    def _sync_optimizer(self):
        super()._sync_optimizer()
        self._infonce_optimizer.sync()
        return

    def _compute_rewards(self):
        """PPO reward is now exclusively driven by the style discriminator."""
        obs = self._exp_buffer.get_data_flat("obs")
        root_pos = self._exp_buffer.get_data_flat("root_pos")
        desired_goal_global = self._exp_buffer.get_data_flat("desired_goal")

        dist_to_goal = torch.norm(desired_goal_global[..., :self._task_dim] - root_pos[..., :self._task_dim], dim=-1)
        is_at_goal = (dist_to_goal < 0.5).float()

        self._update_reward_weights(num_steps_this_iteration=obs.shape[0])

        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        disc_r = self._calc_disc_rewards(norm_disc_obs)

        # Exclusively style component for the reinforcement learning reward
        r = self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        return {
            "disc_reward_mean": disc_reward_mean.item(),
            "active_task_weight": self._task_reward_weight,
            "active_style_weight": self._disc_reward_weight,
            "fitness_mean_dist_to_goal": dist_to_goal.mean().item(),
            "fitness_time_at_goal_rate": is_at_goal.mean().item(),
        }

    def _compute_actor_loss(self, batch):
        """Multi-objective update: PPO Actor Loss + Contrastive Goal Distance."""
        # 1. Compute standard PPO actor loss
        info = super()._compute_actor_loss(batch)

        # 2. Extract batch data for contrastive component
        norm_obs = self._obs_norm.normalize(batch["obs"])
        root_pos = batch["root_pos"]
        desired_goal_global = batch["desired_goal"]
        rand_action_mask = batch["rand_action_mask"]

        # Filter strictly for standard randomized actor samples
        mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[mask]
        root_pos = root_pos[mask]
        desired_goal_global = desired_goal_global[mask]

        # 3. Predict the current actor mode (differentiable wrt actor network)
        a_dist = self._model.eval_actor(norm_obs)
        a_mode = a_dist.mode

        # 4. Construct global state inputs for the encoder
        s_proprio = norm_obs[..., :-self._task_dim]
        # Feed the global position directly into the state-action encoder
        phi_input = torch.cat([s_proprio, root_pos[..., :self._task_dim]], dim=-1)

        # 5. Evaluate the contrastive embeddings
        phi_embed = self._model.eval_phi(phi_input, a_mode)
        psi_embed = self._model.eval_psi(desired_goal_global[..., :self._task_dim])

        # 6. Minimize Euclidean distance between state-action and global goal in embedding space
        dist = torch.sqrt(torch.sum((phi_embed - psi_embed) ** 2, dim=-1))
        contrastive_actor_loss = self._task_reward_weight * torch.mean(dist)

        # 7. Add to the PPO objective
        info["actor_loss"] += contrastive_actor_loss
        info["actor_contrastive_loss"] = contrastive_actor_loss.detach()

        return info

    def _compute_infonce_loss(self, output_current, output_future):
        curr_obs = output_current["curr_obs"]
        curr_action = output_current["curr_action"]
        curr_root_pos = output_current["curr_root_pos"]
        future_root_pos = output_future["curr_root_pos"]

        norm_curr_obs = self._obs_norm.normalize(curr_obs)
        norm_curr_action = self._a_norm.normalize(curr_action)
        s_proprio = norm_curr_obs[..., :-self._task_dim]

        # GLOBAL input structure for environments with obstacles
        phi_input = torch.cat([s_proprio, curr_root_pos[..., :self._task_dim]], dim=-1)
        g_hindsight = future_root_pos[..., :self._task_dim]

        # Pass modified inputs
        phi_embed = self._model.eval_phi(phi_input, norm_curr_action)
        psi_embed = self._model.eval_psi(g_hindsight)

        diff = phi_embed.unsqueeze(1) - psi_embed.unsqueeze(0)
        logits = -torch.sqrt(torch.sum(diff ** 2, dim=-1))

        diag_logits = torch.diag(logits)
        lse = torch.logsumexp(logits + 1e-6, dim=1)

        infonce_loss = -torch.mean(diag_logits - lse)
        lse_penalty = self._logsumexp_penalty_coeff * torch.mean(lse ** 2)
        loss = infonce_loss + lse_penalty

        return {
            "infonce_loss": loss,
            "infonce_base_loss": infonce_loss.detach(),
            "infonce_lse_penalty": lse_penalty.detach(),
            "infonce_logits_mean": logits.mean().detach(),
            "infonce_logits_max": logits.max().detach(),
            "infonce_lse_mean": lse.mean().detach()
        }

    def _reset_done_envs(self, done):
        done_indices = (done != DoneFlags.NULL.value).nonzero(as_tuple=False).flatten().tolist()
        obs, info = super()._reset_done_envs(done)

        for env_idx in done_indices:
            ep_data = self._active_episodes[env_idx]

            if len(ep_data["curr_obs"]) >= 2:
                ep_dict = {
                    "curr_obs": torch.stack(ep_data["curr_obs"]),
                    "curr_action": torch.stack(ep_data["curr_action"]),
                    "curr_root_pos": torch.stack(ep_data["curr_root_pos"]),
                    "curr_root_rot": torch.stack(ep_data["curr_root_rot"]),
                }
                self._traj_buffer.push_episode(ep_dict)

            self._active_episodes[env_idx] = {
                "curr_obs": [], "curr_action": [], "curr_root_pos": [], "curr_root_rot": []
            }
        return obs, info

    def _reset_envs(self, env_ids=None):
        obs, info = super()._reset_envs(env_ids)

        if env_ids is None:
            for env_idx in range(self.get_num_envs()):
                self._active_episodes[env_idx] = {
                    "curr_obs": [], "curr_action": [], "curr_root_pos": [], "curr_root_rot": []
                }
        return obs, info

    def _update_model(self):
        info = super()._update_model()
        infonce_info = dict()

        if self._traj_buffer.is_ready(self._infonce_batch_size):
            for _ in range(self._infonce_epochs):
                output_current, output_future = self._traj_buffer.sample_infonce_pairs(self._infonce_batch_size)
                loss_dict = self._compute_infonce_loss(output_current, output_future)

                loss = loss_dict["infonce_loss"]
                self._infonce_optimizer.step(loss)

                torch_util.add_torch_dict(loss_dict, infonce_info)
            torch_util.scale_torch_dict(1.0 / self._infonce_epochs, infonce_info)

        return {**info, **infonce_info}