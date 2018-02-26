
import numpy as np


''' noise modes '''
UNIFORM = 'uniform'
GAUSSIAN = 'gaussian'

def absolute_noise(noise, size=None):
    mode, bound = noise
    return _generate_noise(mode, bound, size=size)

def percent_noise(quantity, noise, size=None):
    mode, percent = noise
    bound = quantity * percent
    return _generate_noise(mode, bound, size=size)

'''
Select mode and generate noises within the given bound
'''
def _generate_noise(mode, bound, size=None):
    if mode == UNIFORM:
        noise_func = _noise_uniform
    elif mode == GAUSSIAN:
        noise_func = _noise_gaussian
    return noise_func(bound, size=size)

'''
Returns uniform noise in [-bound, +bound]
'''
def _noise_uniform(bound, size=None):
    return np.random.uniform(low=-bound, high=bound, size=size)

'''
Returns (cropped) gaussian noise with mean 0 and std (bound / 3)
(cropped into [-bound, +bound] to pervent negative timestep/rate constants etc.)
'''
def _noise_gaussian(bound, size=None):
    noise = np.random.normal(scale=bound / 3, size=size)
    if size is None:
        noise = max(min(noise, bound), -bound)
    else:
        noise[noise > bound] = bound
        noise[noise < -bound] = -bound
    return noise
