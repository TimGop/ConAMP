import torch
import torch.nn.functional as F
import util.torch_util as torch_util
import learning.clamp_model as clamp_model
import learning.amp_agent as amp_agent
import learning.mp_optimizer as mp_optimizer

import util.torch_util as torch_util
import torch
import torch.nn.functional as F
import util.torch_util as torch_util
import learning.amp_agent as amp_agent
from learning.trajectory_buffer import TrajectoryBuffer
from envs.base_env import DoneFlags
import util.mp_util as mp_util


class CLAMPRewardAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._task_dim = config.get("task_dim", 2)

        # Add penalty coefficient like in 1000 layer rl paper
        self._logsumexp_penalty_coeff = config.get("logsumexp_penalty_coeff", 0.1)

        # k: How many steps to stay purely on AMP (Warmup)
        self._warmup_steps = config.get("warmup_steps", 1_000_000)

        # N: How many steps to take to linearly interpolate the weights (Anneal)
        self._anneal_steps = config.get("anneal_steps", 2_000_000)

        # The final weights you want after the anneal is finished
        self._target_task_weight = config.get("target_task_weight", 0.5)
        self._target_style_weight = config.get("target_style_weight", 0.5)

        # Initialize the active weights to the Warmup Phase limits
        self._task_reward_weight = 0.0
        self._disc_reward_weight = 1.0

        # Track total physical steps for the curriculum
        self._curriculum_total_steps = 0

        # InfoNCE Training Config ---
        self._infonce_epochs = config.get("infonce_epochs", 5)
        self._infonce_batch_size = config.get("infonce_batch_size", 256)
        return

    def _update_reward_weights(self, num_steps_this_iteration):
        """
        Dynamically adjusts the task and style reward weights based on the step count.
        """
        # Get the total environment steps taken so far
        # Increment our internal step counter
        self._curriculum_total_steps += num_steps_this_iteration
        step = self._curriculum_total_steps

        if step <= self._warmup_steps:
            # Phase 1: Pure Warmup (k)
            self._task_reward_weight = 0.0
            self._disc_reward_weight = 1.0

        elif step >= (self._warmup_steps + self._anneal_steps):
            # Phase 3: Mastery (Past k + N)
            self._task_reward_weight = self._target_task_weight
            self._disc_reward_weight = self._target_style_weight

        else:
            # Phase 2: Linear Annealing (Between k and k + N)
            # Calculate how far along we are in the anneal phase (0.0 to 1.0)
            alpha = (step - self._warmup_steps) / self._anneal_steps

            # Interpolate Task Weight (from 0.0 up to Target)
            self._task_reward_weight = alpha * self._target_task_weight

            # Interpolate Style Weight (from 1.0 down to Target)
            self._disc_reward_weight = 1.0 + alpha * (self._target_style_weight - 1.0)

        return

    def _record_data_pre_step(self, obs, info, action, action_info):
        # 1. Let PPO do its standard recording
        super()._record_data_pre_step(obs, info, action, action_info)

        # 2. Extract the global coordinates for time 't'
        root_pos = info["root_pos"]
        root_rot = info["root_rot"]
        desired_goal = info["desired_goal"]

        # 3. Record them in the PPO buffer so _compute_rewards can use them!
        self._exp_buffer.record("root_pos", root_pos)
        self._exp_buffer.record("root_rot", root_rot)
        self._exp_buffer.record("desired_goal", desired_goal)

        # 4. CRITICAL FIX: Append to staging cache using .clone().detach() !
        # Otherwise PyTorch appends a view of the live buffer, meaning
        # all steps in the episode will overwrite themselves to match the final step.
        for env_idx in range(self.get_num_envs()):
            self._active_episodes[env_idx]["curr_obs"].append(obs[env_idx].clone().detach())
            self._active_episodes[env_idx]["curr_action"].append(action[env_idx].clone().detach())
            self._active_episodes[env_idx]["curr_root_pos"].append(root_pos[env_idx].clone().detach())
            self._active_episodes[env_idx]["curr_root_rot"].append(root_rot[env_idx].clone().detach())

        return

    def _build_exp_buffer(self, config):
        # 1. Build the standard PPO and AMP buffers first
        super()._build_exp_buffer(config)

        # 2. Build our new Contrastive Trajectory Buffer
        traj_buffer_size = config.get("traj_buffer_size", 1000)
        max_ep_len = config.get("max_ep_len", 1200)

        # Make sure TrajectoryBuffer is imported at the top of your file!
        self._traj_buffer = TrajectoryBuffer(
            max_episodes=traj_buffer_size,
            max_ep_len=max_ep_len,
            device=self._device,
            gamma=config.get("discount", 0.99)
        )

        # 3. Create the Staging Cache for the parallel environments
        # We need one dictionary of lists for every environment running
        self._active_episodes = [
            {"curr_obs": [], "curr_action": [], "curr_root_pos": [], "curr_root_rot": []}
            for _ in range(self.get_num_envs())
        ]
        return

    def _build_model(self, config):
        model_config = config["model"]
        # Use our new model class instead of AMPModel!
        self._model = clamp_model.CLAMPModel(model_config, self._env)
        return

    def _build_optimizer(self, config):
        # Build Actor, Critic, and Discriminator optimizers
        super()._build_optimizer(config)

        # Build the InfoNCE optimizer
        infonce_config = config["infonce_optimizer"]  # Add this to your config!
        infonce_params = list(self._model.get_infonce_params())
        infonce_params = [p for p in infonce_params if p.requires_grad]

        self._infonce_optimizer = mp_optimizer.MPOptimizer(infonce_config, infonce_params)
        return

    def _sync_optimizer(self):
        # Sync multi-processing gradients
        super()._sync_optimizer()
        self._infonce_optimizer.sync()
        return

    def _compute_rewards(self):
        # Pull rollout data
        obs = self._exp_buffer.get_data_flat("obs")
        action = self._exp_buffer.get_data_flat("action")
        root_pos = self._exp_buffer.get_data_flat("root_pos")
        root_rot = self._exp_buffer.get_data_flat("root_rot")
        desired_goal_global = self._exp_buffer.get_data_flat("desired_goal")

        # NORMALIZE INPUTS HERE
        norm_obs = self._obs_norm.normalize(obs)
        norm_action = self._a_norm.normalize(action)

        dist_to_goal = torch.norm(desired_goal_global[..., :self._task_dim] - root_pos[..., :self._task_dim], dim=-1)
        is_at_goal = (dist_to_goal < 0.5).float()

        # Update weights according to curriculum
        self._update_reward_weights(num_steps_this_iteration=obs.shape[0])

        # Compute InfoNCE Task Reward
        if self._task_reward_weight > 0.0 and self._traj_buffer.is_ready(self._infonce_batch_size):

            # SLICE NORMALIZED OBS
            s_proprio = norm_obs[..., :-self._task_dim]

            rel_pos_global = desired_goal_global - root_pos
            heading_rot_inv = torch_util.calc_heading_quat_inv(root_rot)
            ego_goal_3d = torch_util.quat_rotate(heading_rot_inv, rel_pos_global)
            g_desired = ego_goal_3d[..., :self._task_dim]

            with torch.no_grad():
                # PASS NORMALIZED INPUTS
                phi_embed = self._model.eval_phi(s_proprio, norm_action)
                psi_embed = self._model.eval_psi(g_desired)

                # USE NEGATIVE L2 DISTANCE (No normalization)
                dists = -torch.sqrt(torch.sum((phi_embed - psi_embed) ** 2, dim=-1))

                # Exponential mapping limits reward to [0, 1]
                # (times 10 to match average magnitude of style rewards approx.)
                learned_task_r = torch.exp(dists) * 12  # TODO: maybe add multiplier to config!
        else:
            learned_task_r = torch.zeros(dist_to_goal.shape).to(device=self._device)
            """# WARMUP FALLBACK: Provide a small Euclidean reward to seed the TrajectoryBuffer
            # If we don't do this, the agent never explores, and InfoNCE learns on garbage data
            learned_task_r = torch.exp(-0.5 * dist_to_goal)""" # found to not be necessary

        # Compute AMP Discriminator Reward
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        disc_r = self._calc_disc_rewards(norm_disc_obs)

        """if mp_util.is_root_proc():
            print(
                f"Total Steps: {self._curriculum_total_steps} | Style Reward (Mean): {disc_r.mean().item():.4f} | Min: {disc_r.min().item():.4f} | Max: {disc_r.max().item():.4f}")"""

        # Merge using the dynamically updated weights!
        r = (self._task_reward_weight * learned_task_r) + (self._disc_reward_weight * disc_r)
        self._exp_buffer.set_data_flat("reward", r)

        # Return metrics for the logger
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        return {
            "task_reward_mean": learned_task_r.mean().item(),
            "disc_reward_mean": disc_reward_mean.item(),
            "active_task_weight": self._task_reward_weight,
            "active_style_weight": self._disc_reward_weight,
            "fitness_mean_dist_to_goal": dist_to_goal.mean().item(),
            "fitness_time_at_goal_rate": is_at_goal.mean().item(),
        }

    def _compute_infonce_loss(self, output_current, output_future):
        curr_obs = output_current["curr_obs"]
        curr_action = output_current["curr_action"]
        curr_root_pos = output_current["curr_root_pos"]
        curr_root_rot = output_current["curr_root_rot"]
        future_root_pos = output_future["curr_root_pos"]

        # NORMALIZE INPUTS
        norm_curr_obs = self._obs_norm.normalize(curr_obs)
        norm_curr_action = self._a_norm.normalize(curr_action)
        s_proprio = norm_curr_obs[..., :-self._task_dim]

        rel_pos_global = future_root_pos - curr_root_pos
        heading_rot_inv = torch_util.calc_heading_quat_inv(curr_root_rot)
        ego_future_pos = torch_util.quat_rotate(heading_rot_inv, rel_pos_global)
        g_hindsight = ego_future_pos[..., :self._task_dim]

        # PASS NORMALIZED INPUTS
        phi_embed = self._model.eval_phi(s_proprio, norm_curr_action)
        psi_embed = self._model.eval_psi(g_hindsight)

        # COMPUTE ALL-PAIRS NEGATIVE L2 DISTANCE
        # Creates a [Batch, Batch] matrix of distances
        diff = phi_embed.unsqueeze(1) - psi_embed.unsqueeze(0)
        logits = -torch.sqrt(torch.sum(diff ** 2, dim=-1))

        # INFO NCE LOSS
        diag_logits = torch.diag(logits)
        lse = torch.logsumexp(logits + 1e-6, dim=1)

        # Cross Entropy Formulation: -mean(positive_logit - logsumexp(all_logits))
        infonce_loss = -torch.mean(diag_logits - lse)

        # LOGSUMEXP PENALTY (Critical for unbounded L2 distances!)
        penalty_coeff = self._config.get("logsumexp_penalty_coeff", 0.1)
        lse_penalty = penalty_coeff * torch.mean(lse ** 2)

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
        # find which environments just finished BEFORE calling super()
        # because super() will reset the 'done' tensor to 0 in-place!
        done_indices = (done != DoneFlags.NULL.value).nonzero(as_tuple=False).flatten().tolist()

        # 2. Let the parent class handle the actual environment resets
        obs, info = super()._reset_done_envs(done)

        # 3. Push completed episodes to the TrajectoryBuffer
        for env_idx in done_indices:
            ep_data = self._active_episodes[env_idx]

            # InfoNCE needs at least 2 steps for current/future pairs
            if len(ep_data["curr_obs"]) >= 2:
                # Convert lists of tensors to a single tensor [seq_len, ...]
                ep_dict = {
                    "curr_obs": torch.stack(ep_data["curr_obs"]),
                    "curr_action": torch.stack(ep_data["curr_action"]),
                    "curr_root_pos": torch.stack(ep_data["curr_root_pos"]),
                    "curr_root_rot": torch.stack(ep_data["curr_root_rot"]),
                }
                self._traj_buffer.push_episode(ep_dict)

            # Clear the staging cache for this environment
            self._active_episodes[env_idx] = {
                "curr_obs": [], "curr_action": [], "curr_root_pos": [], "curr_root_rot": []
            }

        return obs, info

    def _update_model(self):
        # Update the PPO Actor/Critic and the AMP Discriminator
        info = super()._update_model()

        infonce_info = dict()

        # Only train InfoNCE if the buffer has enough completed episodes!
        if self._traj_buffer.is_ready(self._infonce_batch_size):
            for _ in range(self._infonce_epochs):
                # Sample a batch of geometrically-spaced positive pairs
                output_current, output_future = self._traj_buffer.sample_infonce_pairs(self._infonce_batch_size)

                # Compute loss
                loss_dict = self._compute_infonce_loss(output_current, output_future)

                # Step the optimizer
                loss = loss_dict["infonce_loss"]
                self._infonce_optimizer.step(loss)

                torch_util.add_torch_dict(loss_dict, infonce_info)

            torch_util.scale_torch_dict(1.0 / self._infonce_epochs, infonce_info)

        # Merge dictionaries for your logger
        return {**info, **infonce_info}