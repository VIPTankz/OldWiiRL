import gym
from gym import spaces
import numpy as np
from collections import deque
import cv2

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84])
        return x_t.astype(np.uint8)

class ProcessFrame10054_2(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame10054_2, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(54, 100, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame10054_2.process(obs)

    @staticmethod
    def process(im):

        im = np.reshape(im, [108, 200, 1]).astype(np.float32)
        x_t = cv2.resize(im, (100, 54), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, [54, 100,1])

        return x_t.astype(np.uint8)

class ProcessFrameUint(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrameUint, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(78, 94, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrameUint.process(obs)

    @staticmethod
    def process(im):

        im = np.reshape(im, [78, 94, 1])#.astype(np.float32)
        #x_t = cv2.resize(im, (100, 54), interpolation=cv2.INTER_AREA)
        #x_t = np.reshape(x_t, [54, 100,1])

        return im.astype(np.uint8)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))




def wrap_env(env, stack_frames=4):

    env = ProcessFrameUint(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, stack_frames)

    return env
