import gym
import numpy as np


## Fire reset wrapper
class FireReset(gym.Wrapper):
    """Fire reset wrapper for gym.Env.
    Take action "fire" on reset.

    Args:
        env (gym.Env): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', (
            'Only use fire reset wrapper for suitable environment!')
        assert len(env.unwrapped.get_action_meanings()) >= 3, (
            'Only use fire reset wrapper for suitable environment!')

    def step(self, action):

        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)

        if done:
            obs = self.env.reset(**kwargs)

        return obs


## Noop reset env
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops

        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101

        assert noops > 0
        obs = None

        for _ in range(noops):

            obs, _, done, _ = self.env.step(self.noop_action)

            if done:
                obs = self.env.reset(**kwargs)

        return obs


## Episodic life env
class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN since it helps value estimation.

    Args:
        env (gym.Env): The environment to be wrapped.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = True
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            self.was_real_done = False
        self.lives = lives

        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.lives = 0

        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()

        return obs


## Max and skip env
class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame.

    Args:
        env (gym.Env): The environment to be wrapped.
        skip (int): The frame mark to be skipped.
    """
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last two observations"""
        total_reward = 0.0
        done = None

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)

            if i == self._skip - 2:
                self._obs_buffer[0] = obs

            elif i == self._skip - 1:
                self._obs_buffer[1] = obs

            total_reward += reward

            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info
