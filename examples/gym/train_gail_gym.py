from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import logging
import os

import chainer
from chainer import functions as F
import gym
import gym.spaces
import gym.wrappers
import numpy as np

import chainerrl

from chainerrl.agents.behavioral_cloning import BehavioralCloning
from chainerrl.agents.gail import Discriminator, GAIL
from chainerrl.misc.expert_dataset import ExpertDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--env', type=str, default='Hopper-v2',
                        help='Gym Env ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--expert-dir', type=str, default='data',
                        help='Dicrectory path storing expert trajectories.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total time steps for training.')
    parser.add_argument('--eval-interval', type=int, default=10000,
                        help='Interval between evaluation phases in steps.')
    parser.add_argument('--eval-n-runs', type=int, default=10,
                        help='Number of episodes ran in an evaluation phase')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run demo episodes, not training')
    parser.add_argument('--load', type=str, default='',
                        help='Directory path to load a saved agent data from'
                             ' if it is a non-empty string.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor the env by gym.wrappers.Monitor.'
                             ' Videos and additional log will be saved.')

    parser.add_argument('--discriminator-lr', type=float, default=3e-4)
    parser.add_argument('--vf-lr', type=float, default=3e-4)
    parser.add_argument('--policy-update-interval', type=int, default=1024,
                        help='Interval steps of TRPO iterations.')
    parser.add_argument('--discriminator-update-interval', type=int,
                        default=3072,
                        help='Interval steps of Discriminator iterations.')
    parser.add_argument('--policy-entropy-coef', type=float, default=0)
    parser.add_argument('--discriminator-entropy-coef', type=float,
                        default=1e-3)

    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Pretrain agents by Behavioral Cloning')
    parser.add_argument('--pretrain-num-epochs', type=int, default=1000)

    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set random seed
    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    if not isinstance(obs_space, gym.spaces.Box):
        print("""\
This example only supports gym.spaces.Box observation spaces. To apply it to
other observation spaces, use a custom phi function that convert an observation
to numpy.ndarray of numpy.float32.""")  # NOQA
        return

    # ================ Set up TRPO ================
    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)

    if isinstance(action_space, gym.spaces.Box):
        # Use a Gaussian policy for continuous action spaces
        policy = \
            chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_space.low.size,
                action_space.low.size,
                min_action=action_space.low,
                max_action=action_space.high,
                bound_mean=True,
                n_hidden_channels=100,
                n_hidden_layers=2,
                mean_wscale=0.01,
                nonlinearity=F.tanh,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            )
    elif isinstance(action_space, gym.spaces.Discrete):
        # Use a Softmax policy for discrete action spaces
        policy = chainerrl.policies.FCSoftmaxPolicy(
            obs_space.low.size,
            action_space.n,
            n_hidden_channels=100,
            n_hidden_layers=2,
            last_wscale=0.01,
            nonlinearity=F.tanh,
        )
    else:
        print("""\
TRPO only supports gym.spaces.Box or gym.spaces.Discrete action spaces.""")  # NOQA
        return

    # Use a value function to reduce variance
    vf = chainerrl.v_functions.FCVFunction(
        obs_space.low.size,
        n_hidden_channels=100,
        n_hidden_layers=2,
        last_wscale=0.01,
        nonlinearity=F.tanh,
    )

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        policy.to_gpu(args.gpu)
        vf.to_gpu(args.gpu)
        obs_normalizer.to_gpu(args.gpu)

    # TRPO's policy is optimized via CG and line search, so it doesn't require
    # a chainer.Optimizer. Only the value function needs it.
    vf_opt = chainer.optimizers.Adam(alpha=args.vf_lr)
    vf_opt.setup(vf)

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        policy.xp.zeros_like(
            policy.xp.array(obs_space.low), dtype=np.float32)[None],
        name='observation')
    chainerrl.misc.draw_computational_graph(
        [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph(
        [vf(fake_obs)], os.path.join(args.outdir, 'vf'))

    actor = chainerrl.agents.TRPO(
        policy=policy,
        vf=vf,
        vf_optimizer=vf_opt,
        obs_normalizer=obs_normalizer,
        update_interval=args.policy_update_interval,
        conjugate_gradient_max_iter=10,
        conjugate_gradient_damping=1e-1,
        gamma=0.995,
        lambd=0.97,
        vf_epochs=5,
        vf_batch_size=128,
        entropy_coef=args.policy_entropy_coef,
    )

    # ================ Pretrain ================
    experts = ExpertDataset(load_dir=args.expert_dir)
    if args.pretrain:
        bc_opt = chainer.optimizers.Adam(alpha=1e-4)
        bc_opt.setup(policy)
        bc = BehavioralCloning(policy, bc_opt, experts)
        for i in range(args.pretrain_num_epochs):
            if i % 10 == 0:
                print('BC iter: {}'.format(i))
            bc.train()

    # ================ Set up Discriminator ================
    obs_normalizer_disc = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)
    if isinstance(action_space, gym.spaces.Box):
        disc_model = Discriminator(obs_space.low.size, action_space.low.size,
                                   obs_normalizer_disc)
    elif isinstance(action_space, gym.spaces.Discrete):
        disc_model = Discriminator(obs_space.low.size, action_space.n,
                                   obs_normalizer_disc)
    disc_opt = chainer.optimizers.Adam(alpha=args.discriminator_lr)
    disc_opt.setup(disc_model.model)

    # ================ Set up GAIL ================
    agent = GAIL(actor, vf_opt, disc_model, disc_opt, experts,
                 update_interval=args.discriminator_update_interval,
                 discriminator_entropy_coef=args.discriminator_entropy_coef,
                 gpu=args.gpu)

    if args.load:
        agent.load(args.load)

    # ================================

    if args.demo:
        env = make_env(test=True)
        eval_stats = chainerrl.experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:

        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=make_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            train_max_episode_len=timestep_limit,
        )


if __name__ == '__main__':
    main()
