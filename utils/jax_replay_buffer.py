
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from functools import partial

# JAX Replay BUffer
class BufferState(NamedTuple):
    observations: jnp.ndarray
    next_observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    pos: int
    buffer_size: int
    full: bool

class BufferSample(NamedTuple):
    observations: jnp.ndarray
    next_observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

class ReplayBuffer:
    # @staticmethod
    def init(self, buffer_size, observation_space, action_space, device="cuda", n_envs=1):
        self.obs_shape = observation_space.shape
        self.action_dim = np.prod(action_space.shape)
        self.n_envs = n_envs
        self.buffer_size = max(buffer_size // n_envs, 1)
        return self.reset()
    
    def reset(self):
        buffer_state = BufferState(
            observations=jnp.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=jnp.float32),
            next_observations=jnp.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=jnp.float32),
            actions=jnp.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=jnp.float32),
            rewards=jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32),
            dones=jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32),
            pos=0,
            buffer_size=self.buffer_size,
            full=False,
        )
        return buffer_state
    
    @partial(jax.jit, static_argnames=('self',))
    def add(self, buffer_state, obs, next_obs, action, reward, done, infos):
        pos = buffer_state.pos
        new_buffer_state = BufferState(
            observations=buffer_state.observations.at[pos].set(jnp.array(obs)),
            next_observations=buffer_state.next_observations.at[pos].set(jnp.array(next_obs)),
            actions=buffer_state.actions.at[pos].set(jnp.array(action).reshape(self.n_envs, self.action_dim)),
            rewards=buffer_state.rewards.at[pos].set(jnp.array(reward)),
            dones=buffer_state.dones.at[pos].set(jnp.array(done)),
            full=buffer_state.full | (buffer_state.pos + 1 == buffer_state.buffer_size),
            pos=(buffer_state.pos + 1) % buffer_state.buffer_size,
            buffer_size=buffer_state.buffer_size,
        )
        return new_buffer_state

    @partial(jax.jit, static_argnames=('self','batch_size',))
    def sample(self, buffer_state, batch_size, key):
        key, _key = jax.random.split(key)
        # batch_ind = jax.lax.cond(
        #     buffer_state.full,
        #     lambda : jax.random.randint(_key, (batch_size,), 1, buffer_state.buffer_size) + ,
        #     lambda : jax.random.randint(_key, (batch_size,), 0, buffer_state.pos),
        # )

        # Simple sample because we're not optimizing memory
        upper_bound = jax.lax.select(buffer_state.full, buffer_state.buffer_size, buffer_state.pos)
        batch_ind = jax.random.randint(_key, (batch_size,), 0, upper_bound)
        
        key, _key = jax.random.split(key)
        env_ind = jax.random.randint(_key, (batch_size,), 0, self.n_envs)
        
        data = BufferSample(
            observations=buffer_state.observations[batch_ind, env_ind, :],
            next_observations=buffer_state.next_observations[batch_ind, env_ind, :],
            actions=buffer_state.actions[batch_ind, env_ind, :],
            rewards=buffer_state.rewards[batch_ind, env_ind],
            dones=buffer_state.dones[batch_ind, env_ind],
        )
        return data
