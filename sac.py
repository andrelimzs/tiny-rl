import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
# from stable_baselines3.common.buffers import ReplayBuffer
from utils.jax_replay_buffer import ReplayBuffer, BufferState, BufferState
from torch.utils.tensorboard import SummaryWriter
import wandb
from jax.experimental import io_callback
from functools import partial
import distrax
from jax_tqdm import scan_tqdm

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "sjrl-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_name: Optional[str] = None
    """the wandb run name"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    
    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 15e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    test_frequency: int = 10_000_000
    """the frequency of test evaluations"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}",
                episode_trigger=lambda x : (x % 1000) == 0,
                # episode_trigger=lambda x : True,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5*(LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
    
    def sample_action(self, mean, log_std, key):
        std = jnp.exp(log_std)
        normal = distrax.Normal(mean, std)
        x_t, log_prob = normal.sample_and_log_prob(seed=key)
        y_t = jnp.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Action Bounds
        log_prob -= jnp.log(self.action_scale * (1 - jnp.pow(y_t,2) + 1e-6))
        log_prob = log_prob.sum(1, keepdims=True)
        mean = jnp.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def mean_action(self, mean):
        return jnp.tanh(mean) * self.action_scale + self.action_bias

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

def gym_env_step(actions, global_step):
    """Call non-JAX environments via io_callback"""
    def _step(actions, global_step):
        next_obs, rewards, terminations, truncations, raw_infos = envs.step(actions)
        
        # Compute real_next_obs for buffer
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = raw_infos["final_observation"][idx]
        
        # Extract episode statistics
        infos = { "r": 0.0, "l": -1 }
        if "final_info" in raw_infos:
            for info in raw_infos["final_info"]:
                infos["r"] = info["episode"]["r"].item()
                infos["l"] = info["episode"]["l"].item()
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        return next_obs, real_next_obs, rewards, terminations, truncations, infos
    
    return io_callback(_step, env_result_shape, actions, global_step)

def find_env_return_shape(envs, seed):
    """Get shape of gym environment return"""
    envs.reset(seed=seed)
    actions = jnp.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    next_obs, rewards, terminations, truncations, raw_infos = envs.step(actions)

    # Get pytree shape of env step output
    infos = {"r": 0.0, "l": 0}
    env_result = (next_obs, next_obs, rewards, terminations, truncations, infos)
    env_result_shape = jax.tree_util.tree_map(jax.eval_shape, lambda : env_result)
    action_shape = jax.tree_util.tree_map(jax.eval_shape, lambda : actions)
    return env_result_shape, action_shape

def make_agent(envs, key):
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)
    return key, (actor, actor_state), (qf, qf1_state, qf2_state) 

def update_critic(
    actor_state: TrainState,
    qf1_state: TrainState,
    qf2_state: TrainState,
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    terminations: np.ndarray,
    key: jnp.ndarray,
):
    key, _key = jax.random.split(key, 2)
    next_state_actions, next_state_log_pi, _ = actor.sample_action(
        *actor.apply(actor_state.params, next_observations), _key)
    qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
    qf2_next_target = qf.apply(qf2_state.target_params, next_observations, next_state_actions).reshape(-1)
    min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target) - args.alpha * next_state_log_pi.reshape(-1)
    next_q_value = (rewards + (1 - terminations) * args.gamma * (min_qf_next_target)).reshape(-1)

    def mse_loss(params):
        qf_a_values = qf.apply(params, observations, actions).squeeze()
        return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

    (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
    (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
    qf1_state = qf1_state.apply_gradients(grads=grads1)
    qf2_state = qf2_state.apply_gradients(grads=grads2)

    return (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key

def update_actor(
    actor_state: TrainState,
    qf1_state: TrainState,
    qf2_state: TrainState,
    observations: np.ndarray,
    key: jnp.ndarray,
):
    def actor_loss(params, key):
        actions, log_pi, _ = actor.sample_action(*actor.apply(params, observations), key)
        qf1_pi = qf.apply(qf1_state.params, observations, actions)
        qf2_pi = qf.apply(qf2_state.params, observations, actions)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
        return (args.alpha * log_pi - min_qf_pi).mean()
    
    for _ in range(args.policy_frequency):
        key, _key = jax.random.split(key)
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params, _key)
        actor_state = actor_state.apply_gradients(grads=grads)
        # actor_state = actor_state.replace(
        #     target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
        # )

    return actor_state, actor_loss_value

def update_targets(qf1_state, qf2_state):
    qf1_state = qf1_state.replace(
        target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
    )
    qf2_state = qf2_state.replace(
        target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, args.tau)
    )
    return qf1_state, qf2_state

def get_action(global_step, actor_state, obs, key):
    def get_action1(actor_state, obs, key):
        def sample_action():
            return jnp.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        return io_callback(sample_action, action_shape)
    
    def get_action2(actor_state, obs, key):
        actions, _, _ = actor.sample_action(*actor.apply(actor_state.params, obs), key)
        return actions

    return jax.lax.cond(
        global_step < args.learning_starts,
        get_action1, get_action2,
        actor_state, obs, key
    )

def log_fn(losses, global_step):
    for label, value in losses.items():
        writer.add_scalar(f"losses/{label}", value.item(), global_step)
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

def update_network_fn(global_step, network_state, rb_state, key):
    def update_network(network_state, rb_state, key):
        actor_state, (qf1_state, qf2_state) = network_state
        key, _key = jax.random.split(key)
        data = rb.sample(rb_state, args.batch_size, _key)

        key, _key = jax.random.split(key)
        (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
            actor_state,
            qf1_state,
            qf2_state,
            jnp.array(data.observations),
            jnp.array(data.actions),
            jnp.array(data.next_observations),
            jnp.array(data.rewards.flatten()),
            jnp.array(data.dones.flatten()),
            _key,
        )

        # Update actor
        key, _key = jax.random.split(key)
        actor_state, actor_loss_value = jax.lax.cond(
            global_step % args.policy_frequency == 0,
            update_actor,
            lambda a,b,c,d,e: (a, np.nan),
            actor_state, qf1_state, qf2_state, data.observations, _key
        )

        # Update target networks
        qf1_state, qf2_state = jax.lax.cond(
            global_step % args.target_network_frequency == 0,
            update_targets,
            lambda a,b : (a,b),
            qf1_state, qf2_state
        )

        # Logging
        losses = {
            "qf1_loss": qf1_loss_value,
            "qf2_loss": qf2_loss_value,
            "qf1_values": qf1_a_values,
            "qf2_values": qf2_a_values,
            "actor_loss": actor_loss_value,
        }
        jax.lax.cond(
            global_step % 100 == 0,
            lambda x,y : io_callback(log_fn, None, x,y),
            lambda x,y : None,
            losses, global_step
        )

        network_state = actor_state, (qf1_state, qf2_state)
        return network_state
    
    return jax.lax.cond(
        global_step > args.learning_starts,
        update_network,
        lambda a,b,c: a,
        network_state, rb_state, key
    )

def train_step(train_state, global_step):
    (network_state, rb_state, obs, key) = train_state
    actor_state, (qf1_state, qf2_state) = network_state

    # Compute agent action
    key, _key = jax.random.split(key)
    actions = get_action(global_step, actor_state, obs, _key)

    # Simulate environment and update replay buffer
    next_obs, real_next_obs, rewards, terminations, truncations, infos = gym_env_step(actions, global_step)
    rb_state = rb.add(rb_state, obs, real_next_obs, actions, rewards, terminations, infos)
    obs = next_obs

    # Update agent networks
    key, _key = jax.random.split(key)
    network_state = update_network_fn(global_step, network_state, rb_state, _key)
    
    # Test performance
    key, _key = jax.random.split(key)
    test_fn(global_step, network_state, _key)

    return (network_state, rb_state, obs, key), global_step + 1

def train(train_state):
    return jax.lax.scan(
        scan_tqdm(args.total_timesteps, 1000)(train_step),
        train_state,
        jnp.arange(args.total_timesteps)
    )

def test_fn(global_step, network_state, key):
    def test(global_step, network_state, key):
        actor_state, (qf1_state, qf2_state) = network_state

        # key, _key = jax.random.split(key)
        test_obs, _ = test_envs.reset(seed=args.seed)

        # Simulate until termination
        while True:
            mean, log_std = actor.apply(actor_state.params, test_obs)
            actions = actor.mean_action(mean)
            test_obs, _, _, _, test_infos = test_envs.step(actions)
            if "final_info" in test_infos:
                break
        
        # Log test resutls
        for info in test_infos["final_info"]:
            writer.add_scalar("test/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("test/episodic_length", info["episode"]["l"], global_step)
    
    return jax.lax.cond(
        (global_step > args.learning_starts) & (global_step % args.test_frequency == 0),
        partial(io_callback, test, None),
        lambda *x : None,
        global_step, network_state, key
    )

if __name__ == "__main__":
    # Colorize output
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

    # Parse arguments
    args = tyro.cli(Args)
    if args.wandb_name is None:
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.wandb_name
    
    # Track training
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Set up environments
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    test_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Set up replay buffer
    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer()
    rb_state = rb.init(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
    )

    # Reset environment and make agent
    env_result_shape, action_shape = find_env_return_shape(envs, args.seed)
    obs, _ = envs.reset(seed=args.seed)
    key, (actor, actor_state), (qf, qf1_state, qf2_state) = make_agent(envs, key)
    
    # Compile
    train_jit = jax.jit(train)
    
    # Train
    start_time = time.time()
    network_state = actor_state, (qf1_state, qf2_state)
    train_state = (network_state, rb_state, obs, key)
    train_state, _ = train_jit(train_state)

    # Save model
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                        qf2_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
