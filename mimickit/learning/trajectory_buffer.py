import torch


class TrajectoryBuffer:
    def __init__(self, max_episodes, max_ep_len, device, gamma=0.99):
        """
        Args:
            max_episodes: Maximum number of full trajectories to store.
            max_ep_len: Maximum length of an episode (for tensor pre-allocation).
            device: torch device.
            gamma: Discount factor for geometric future sampling (matches the JAX code).
        """
        self._max_episodes = max_episodes
        self._max_ep_len = max_ep_len
        self._device = device
        self._gamma = gamma

        self._buffers = dict()
        self._ep_lengths = torch.zeros(max_episodes, dtype=torch.long, device=device)

        self._head = 0
        self._size = 0

    def add_buffer(self, name, data_shape, dtype):
        """Pre-allocates a 3D tensor: [Episode, Time, *Shape]"""
        assert name not in self._buffers
        buffer_shape = [self._max_episodes, self._max_ep_len] + list(data_shape)
        self._buffers[name] = torch.zeros(buffer_shape, dtype=dtype, device=self._device)

    def push_episode(self, ep_data_dict):
        """
        Pushes a single completed episode into the buffer.
        ep_data_dict: dict of tensors with shape [seq_len, *data_shape]
        """
        seq_len = next(iter(ep_data_dict.values())).shape[0]

        # InfoNCE needs at least 2 steps (a current state and a future state)
        if seq_len < 2:
            return

            # Truncate if the episode somehow exceeds the max length constraint
        seq_len = min(seq_len, self._max_ep_len)

        # Auto-initialize buffers if this is the first push
        if len(self._buffers) == 0:
            for key, data in ep_data_dict.items():
                self.add_buffer(key, data.shape[1:], data.dtype)

        # Store the episode data
        for key, data in ep_data_dict.items():
            self._buffers[key][self._head, :seq_len] = data[:seq_len]

        self._ep_lengths[self._head] = seq_len

        self._head = (self._head + 1) % self._max_episodes
        self._size = min(self._size + 1, self._max_episodes)

    def sample_infonce_pairs(self, batch_size):
        """
        Efficient Hindsight Relabeling:
        Instead of 1 pair per episode, we treat the buffer as a flat pool of transitions
        and sample a future goal for every single sampled transition.
        """
        assert self._size > 0, "Buffer is empty"

        # 1. Sample random episodes
        ep_idx = torch.randint(0, self._size, (batch_size,), device=self._device)
        lengths = self._ep_lengths[ep_idx]

        # 2. Sample random valid 'current' timesteps t in [0, length - 2]
        # (Must leave at least 1 step for the future)
        t = (torch.rand(batch_size, device=self._device) * (lengths - 1)).long()

        # 3. Vectorized Geometric Future Sampling
        # Find the maximum possible future steps needed in this batch
        max_remaining = (lengths - t).max().item()

        # Create a time grid: [batch_size, max_remaining]
        # e.g., row i represents [t_i, t_i+1, t_i+2, ...]
        steps_forward = torch.arange(max_remaining, device=self._device).unsqueeze(0).expand(batch_size, -1)

        # Which of these future steps actually exist in the episode?
        valid_future_mask = steps_forward < (lengths - t).unsqueeze(1)

        # Geometric discount: gamma^(steps_forward)
        probs = valid_future_mask.float() * (self._gamma ** steps_forward.float())

        # The probability of picking the *current* state as a goal should be 0
        # (InfoNCE needs a strictly future goal)
        probs[:, 0] = 0.0

        # Sample the exact number of steps to look forward based on the probabilities
        sampled_forward_steps = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Final future timestep
        t_future = t + sampled_forward_steps

        # 4. Extract and return the paired dictionaries
        output_current = {k: v[ep_idx, t] for k, v in self._buffers.items()}
        output_future = {k: v[ep_idx, t_future] for k, v in self._buffers.items()}

        return output_current, output_future

    def is_ready(self, batch_size):
        return self._size >= (batch_size * 4)