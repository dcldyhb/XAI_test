# utils.py
import copy
import torch

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight tau)
    target = tau*source + (1 - tau)*target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    Copy network parameters from source to target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)