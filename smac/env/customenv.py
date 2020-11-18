from .multiagentenv import MultiAgentEnv
import numpy as np

class MyEnv(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        self.n_agents = 3
        self.episode_limit = 10
        self._i = 0
        self.__state = np.zeros((self.n_agents, 1))
        self.__obs = self.__state
        self.n_actions = self.n_agents

    def step(self, actions): # returns reward, terminated, info
        for i, a in enumerate(actions):
            self.__state[i] += a
        self._i += 1
        return self.__state, [False] * self.n_agents, {}

    def get_obs(self): # return observation for each agent in a list:
        return [i for i in self.__obs.ravel()]

    def get_obs_agent(self, agent_id):
        return self.__obs.ravel()[agent_id]

    def get_obs_size(self): # return observation size for 1 agent:
        return self.__obs.shape[1]

    def get_state(self): # return the system state
        return self.__state

    def get_state_size(self): # return the system state size
        return self.__state.ravel().shape[0]

    def get_avail_actions(self): # return possible actions
        return [np.arange(self.n_agents) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id): # return possible actions for an agent
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self): #
        return self.n_agents

    def reset(self):
        self._i = 0
        self.__state = np.zeros(self.n_agents)
        self.__obs = self.__state

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass
