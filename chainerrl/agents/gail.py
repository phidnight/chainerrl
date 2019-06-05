from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainerrl.links import MLP
from chainerrl.agent import AttributeSavingMixin, BatchAgent


class Discriminator(chainer.Chain, AttributeSavingMixin):
    saved_attributes = ('model', 'obs_normalizer')

    def __init__(self, obs_size, acs_size, obs_normalizer=None,
                 hidden_sizes=(100, 100), last_wscale=0.01):
        super().__init__()
        self.obs_size = obs_size
        self.obs_normalizer = obs_normalizer
        self.model = MLP(obs_size + acs_size, 1, hidden_sizes,
                         last_wscale=last_wscale, nonlinearity=F.tanh)

    def __call__(self, data, update_obs_normalizer=False):
        obs = data[:, :self.obs_size]
        acs = data[:, self.obs_size:]
        obs = self.obs_normalizer(
                obs,
                update=update_obs_normalizer)
        data = np.concatenate((obs, acs), axis=1)
        raw_output = self.model(data)
        return F.sigmoid(raw_output)


class GAIL(AttributeSavingMixin, BatchAgent):
    """Generative Adversarial Imination Learning
    See https://arxiv.org/abs/1606.03476
    Args:
        actor (chainerrl.Agent): Policy
        actor_optimizer (chainer.Optimizer): Optimizer to train actor
        discriminator (Discriminator): Discriminator
        discriminator_optimizer (chainer.Optimizer): Optimizer to train
            discriminator
        experts (ExpertDataset): Expert trajectory
        update_interval (int): Interval steps of discriminator iterations.
            Every after this amount of steps, this agent updates
            the discriminator using data from these steps and experts.
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        entropy_coef (float): Weight coefficient of discriminator for entropy
            bonus [0, inf)
        discriminator_loss_decay (float): Decay rate of average loss of
            discriminator, only used for recording statistics
        discriminator_entropy_decay (float): Decay rate of average entropy of
            discriminator, only used for recording statistics
        gpu (int): GPU device id if not None nor negative
    """
    saved_attributes = ('actor', 'actor_optimizer',
                        'discriminator', 'discriminator_optimizer')

    def __init__(self, actor, actor_optimizer,
                 discriminator, discriminator_optimizer,
                 experts, update_interval=1024,
                 minibatch_size=3072, epochs=1,
                 discriminator_entropy_coef=1e-3,
                 discriminator_loss_decay=0.99,
                 discriminator_entropy_decay=0.99,
                 gpu=None):
        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.experts = experts
        self.discriminator_input_dim = (
            self.experts.obs.shape[1] + self.experts.action.shape[1])
        self._reset_trajectories()
        self.last_episode = []
        self.last_observation = None
        self.last_action = None
        self.last_discriminator_value = 0
        self.t = 0
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.discriminator_entropy_coef = discriminator_entropy_coef
        self.discriminator_loss_decay = discriminator_loss_decay
        self.discriminator_entropy_decay = discriminator_entropy_decay
        self.discriminator_average_loss = 0.
        self.accuracy_gen = 0.
        self.accuracy_exp = 0.
        self.discriminator_entropy = 1.
        self.update_count = 0

        self.batch_last_state = None
        self.batch_last_action = None

    def _reset_trajectories(self):
        self.trajectories = np.empty((0, self.discriminator_input_dim),
                                     'float32')

    def _get_entropy(self, values):
        return F.average((-values * F.log2(values + 1e-8)
                         - (1 - values) * F.log2(1 - values + 1e-8)))

    def _loss(self, fake_batch, true_batch):
        infer_fake = self.discriminator(fake_batch)
        L1 = -F.average(F.log(1 - infer_fake + 1e-8))
        infer_true = self.discriminator(true_batch, update_obs_normalizer=True)
        L2 = -F.average(F.log(infer_true + 1e-8))
        entropy = (
            self._get_entropy(infer_fake) / 2
            + self._get_entropy(infer_true) / 2)
        loss = L1 + L2 - entropy * self.discriminator_entropy_coef
        # Update stats
        self.discriminator_average_loss *= self.discriminator_loss_decay
        self.discriminator_average_loss += (
            (1.0 - self.discriminator_loss_decay) * loss.array)
        self.accuracy_gen = np.average(infer_fake.array < 0.5)
        self.accuracy_exp = np.average(infer_true.array > 0.5)
        self.discriminator_entropy = (
            self.discriminator_entropy_decay * self.discriminator_entropy
            + (1. - self.discriminator_entropy_decay) * entropy.array)
        return loss

    def _update(self):
        trajectory_size = len(self.trajectories)
        expert_selected_dataset = self.experts.get_samples(trajectory_size)
        fake_iter = chainer.iterators.SerialIterator(
            self.trajectories, self.minibatch_size)
        true_iter = chainer.iterators.SerialIterator(
            expert_selected_dataset, self.minibatch_size)
        while fake_iter.epoch < self.epochs:
            fake_batch = np.array(fake_iter.__next__())
            true_batch = np.array(true_iter.__next__())
            self.discriminator_optimizer.update(
                lambda: self._loss(fake_batch, true_batch))

    def _update_if_dataset_is_ready(self):
        if len(self.trajectories) >= self.update_interval:
            self._update()
            self.update_count += 1
            self._reset_trajectories()

    def act(self, obs):
        return self.actor.act(obs)

    def act_and_train(self, obs, reward):
        action = self.actor.act_and_train(obs, self.last_discriminator_value)
        self.t += 1
        self.last_observation = obs
        self.last_action = action
        obs_and_action = np.concatenate((obs, action), axis=0)
        discriminator_input = np.expand_dims(obs_and_action, axis=0)
        self.last_discriminator_value = -F.log(1 - self.discriminator(discriminator_input) + 1e-8).array[0, 0]  # NOQA
        self.trajectories = np.append(self.trajectories,
                                      discriminator_input, axis=0)
        self._update_if_dataset_is_ready()
        return action

    def stop_episode_and_train(self, obs, reward, done):
        assert self.last_observation is not None
        assert self.last_action is not None

        self.t += 1
        self.actor.stop_episode_and_train(obs,
                                          self.last_discriminator_value,
                                          done)
        self.stop_episode()

    def stop_episode(self):
        self.last_observation = None
        self.last_action = None
        self.last_discriminator_value = 0
        self.actor.stop_episode()

    def batch_act(self, batch_obs):
        batch_action = self.actor.batch_act(batch_obs)
        return batch_action

    def batch_act_and_train(self, batch_obs):
        batch_action = self.actor.batch_act_and_train(batch_obs)
        self.batch_last_state = batch_obs
        self.batch_last_action = batch_action
        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):
        # update policy
        batch_discriminator_values = []
        for obs, action in zip(self.batch_last_state, self.batch_last_action):
            obs_and_action = np.concatenate((obs, action), axis=0)
            discriminator_input = np.expand_dims(obs_and_action, axis=0)
            reward = -F.log(1 - self.discriminator(discriminator_input) + 1e-8).array[0, 0]  # NOQA
            batch_discriminator_values.append(reward)

        self.actor.batch_observe_and_train(batch_obs,
                                           tuple(batch_discriminator_values),
                                           batch_done, batch_reset)

        # update discriminator
        for obs, action in zip(self.batch_last_state, self.batch_last_action):
            obs_and_action = np.concatenate((obs, action), axis=0)
            discriminator_input = np.expand_dims(obs_and_action, axis=0)
            self.trajectories = np.append(self.trajectories,
                                          discriminator_input, axis=0)
            self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
           ('accuracy_gen', self.accuracy_gen),
           ('accuracy_exp', self.accuracy_exp),
           ('discriminator_average_loss', self.discriminator_average_loss),
           ('discriminator_entropy', self.discriminator_entropy)
        ] + [('actor_' + name, loss)
             for name, loss in self.actor.get_statistics()]
