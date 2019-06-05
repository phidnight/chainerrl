from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import numpy as np


class ExpertDataset:
    def __init__(self, load_dir=None):
        self.obs = None
        self.action = None
        self.next_obs = None
        self.nonterminal = None
        self.size = 0
        if load_dir is not None:
            self.load(load_dir)

    def append(self, obs, action, next_obs, nonterminal):
        self.obs = obs if self.obs is None else np.vstack((self.obs, obs))
        self.action = (
            action if self.action is None else
            np.vstack((self.action, action)))
        self.next_obs = (
            next_obs if self.next_obs is None else
            np.vstack((self.next_obs, next_obs)))
        self.nonterminal = (
            nonterminal if self.nonterminal is None else
            np.vstack((self.nonterminal, nonterminal)))
        self.size += 1

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'experts.npz'),
                 obs=self.obs, action=self.action,
                 next_obs=self.next_obs, nonterminal=self.nonterminal)

    def load(self, load_dir):
        data = np.load(os.path.join(load_dir, 'experts.npz'))
        self.obs = data['obs']
        self.action = data['action']
        self.next_obs = data['next_obs']
        self.nonterminal = data['nonterminal']
        self.size = len(self.obs)

    def get_samples(self, sample_size):
        keys = np.random.permutation(self.size)[:sample_size]
        obs = self.obs[keys]
        action = self.action[keys]
        return np.concatenate((obs, action), axis=1)
