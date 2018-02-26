
import numpy as np
import gym.spaces
from .atrp_base import ATRPBase, MONO, DORM, MARGIN_SCALE


'''
ATRP simulation environment aiming at achieving a target distribution.
    Target is considered achieved if the maximum difference in pdf is less than
    a given threshold. +1 reward if less than `thres_loose`, +2 if less than
    `thres_tight`.

Input arguments:
    reward_chain_type: type of chain that the reward is related with;
    target:            target distribution (of the rewarding chain type);
    thres_loose:       loose threshold for agreement of distributions;
    thres_tight:       tight threshold for agreement of distributions.
'''

class ATRPTargetDistribution(ATRPBase):

    def _init_reward(self, reward_chain_type=DORM, target=None,
                     thres_loose=5e-3, thres_tight=2e-3,
                     reward_loose=0.1, reward_tight=1.0):
        reward_chain_type = reward_chain_type.lower()
        self.reward_chain_type = reward_chain_type
        self.target = target = np.array(target)
        self.thres_loose = thres_loose
        self.thres_tight = thres_tight
        self.reward_loose = reward_loose
        self.reward_tight = reward_tight
        target_mono_quant = target.dot(np.arange(1, 1 + len(target)))
        self.target_quant = target / target_mono_quant * self.add_cap[MONO]
        self.target_ymax = np.max(self.target_quant) * MARGIN_SCALE

    def _reward(self, done):
        if done:
            chain = self.ending_chain = self.chain(self.reward_chain_type)
            target = self.target
            target = target / np.sum(target)
            current = chain / np.sum(chain)
            max_diff = np.max(np.abs(target - current))
            reward = 0.0
            if max_diff < self.thres_loose:
                reward = self.reward_loose
            if max_diff < self.thres_tight:
                reward = self.reward_tight
        else:
            reward = 0.0
        return reward

    def _render_reward_init(self, key, axis):
        if key == self.reward_chain_type:
            target_quant = self.target_quant
            target_label = 'Target distribution'
            len_values = len(target_quant)
            linspace = np.linspace(1, len_values, len_values)
            axis.plot(linspace, target_quant, 'r', label=target_label)
            self._render_reward_update(key, axis) # reuse to set ylim

    def _render_reward_update(self, key, axis):
        if key == self.reward_chain_type:
            _, ymax = axis.get_ylim()
            if ymax < self.target_ymax:
                axis.set_ylim([0, self.target_ymax])

