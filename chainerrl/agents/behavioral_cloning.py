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
from chainerrl.agent import AttributeSavingMixin, BatchAgent


class BehavioralCloning(AttributeSavingMixin, BatchAgent):
    """Behavioral Cloning
    Args:
        model (A3CModel): Model
        optimizer (chainer.Optimizer): Optimizer to train model
        experts (ExpertDataset): Expert trajectory
        minibatch_size (int): Minibatch size
        states_per_epoch (int): Number of states to use in one training
            iteration
        gpu (int): GPU device id if not None nor negative
    """
    saved_attributes = ('model', 'optimizer')

    def __init__(self, model, optimizer,
                 experts, minibatch_size=128,
                 states_per_epoch=2048, gpu=None):
        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)

        self.model = model
        self.optimizer = optimizer
        self.experts = experts
        self.minibatch_size = minibatch_size
        self.states_per_epoch = states_per_epoch

    def act(self, obs):
        return self.model.act(obs)

    def act_and_train(self, obs, reward):
        raise NotImplementedError

    def stop_episode_and_train(self, obs, reward, done):
        raise NotImplementedError

    def stop_episode(self):
        pass

    def batch_act(self, batch_obs):
        raise NotImplementedError

    def batch_act_and_train(self, batch_obs):
        raise NotImplementedError

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):
        raise NotImplementedError

    def _loss(self, batch_obs, batch_acs):
        action = self.model(batch_obs).sample()
        return F.mean_squared_error(action, batch_acs)

    def train(self, epochs=1):
        expert_selected_obs, expert_selected_acs = self.experts.get_samples(
            self.states_per_epoch)
        data_iter = chainer.iterators.SerialIterator(
            np.random.permutation(np.arange(self.states_per_epoch)),
            self.minibatch_size)
        while data_iter.epoch < epochs:
            batch_keys = np.array(data_iter.__next__())
            batch_obs = expert_selected_obs[batch_keys]
            batch_acs = expert_selected_acs[batch_keys]
            self.optimizer.update(
                lambda: self._loss(batch_obs, batch_acs))

    def get_statistics(self):
        return [('average_loss', self.average_loss)]
