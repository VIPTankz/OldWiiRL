import numpy as np


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=1.0,eps_dec = 1e-6,eps_min = 0.05, selector=None):
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        self.epsilon -= self.eps_dec
        if self.epsilon < self.eps_min:
            self.epsilon = self.eps_min
        return actions

class StickyEpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=1.0,eps_dec = 1e-6,eps_min = 0.05, selector=None):
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.selector = selector if selector is not None else ArgmaxActionSelector()
        self.repeat_probs = 0.3
        self.prev_actions = None

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape

        if np.random.random() < self.repeat_probs:
            if self.prev_actions is not None and batch_size == len(self.prev_actions):
                return self.prev_actions
        
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        self.epsilon -= self.eps_dec
        if self.epsilon < self.eps_min:
            self.epsilon = self.eps_min

        self.prev_actions = actions[:]
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)
