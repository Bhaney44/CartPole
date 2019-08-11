from spinup import ppo
import tensorflow as tf
import gym
env_fn = lambda : gym.make('CartPole-v0')
ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='output_dir_path', exp_name='CartPole_0')
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=25, logger_kwargs=logger_kwargs)
