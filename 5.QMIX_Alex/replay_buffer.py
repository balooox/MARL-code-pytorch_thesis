import numpy as np

class ReplayBuffer():
    def __init__(self, args) -> None:
        self.n_agents = args.n_agents # Number of agents
        self.obs_shape = args.obs_shape # Shape of local (agent) observation
        self.state_shape = args.state_shape # Shape of global state
        self.n_actions = args.n_actions
        self.episode_limit = args.episode_limit # Maximal possible steps per episode
        self.buffer_size = args.buffer_size # Total size of replay buffer
        self.batch_size = args.batch_size # Amount of experience drawn from one lerning step
        self.episode_num = 0
        self.current_size = 0
        self.buffer = {
            "obs_n" : np.zeros((self.buffer_size, self.episode_limit + 1, self.n_agents, self.obs_shape)),
            "s" : np.zeros((self.buffer_size, self.episode_limit +1, self.state_shape)),
            "avail_a_n": np.zeros((self.buffer_size, self.episode_limit + 1, self.n_agents, self.n_actions)),
            "last_onehot_a_n": np.zeros((self.buffer_size, self.episode_limit + 1, self.n_agents, self.n_actions)),
            "a_n": np.zeros((self.buffer_size, self.episode_limit + 1, self.n_agents)),
            "r": np.zeros((self.buffer_size, self.episode_limit + 1, 1)),
            "dw": np.zeros((self.buffer_size, self.episode_limit + 1, 1)),
            "active": np.zeros((self.buffer_size, self.episode_limit + 1 , 1))
        }
        self.episode_len = np.zeros(self.buffer_size)

    def store_transition(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0
        
        

    def store_last_item(self, episode_step, obs_n, s, avail_a_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n 

        self.episode_len[self.episode_num] = episode_step
        self.episode_num = (self.episode_num + 1) % self.batch_size  # after reaching buffer size, episode_num gets reseted to zero, implementing a deque 
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        
        indices = np.random.choice(range(self.current_size), size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[indices]))
        print(indices)

        buffer = {
            "obs_n" : self.buffer['obs_n'][indices],
            "s" : self.buffer['s'][indices],
            "avail_a_n": self.buffer['avail_a_n'][indices],
            "last_onehot_a_n": self.buffer['last_onehot_a_n'][indices],
            "a_n": self.buffer['a_n'][indices],
            "r": self.buffer['r'][indices],
            "dw": self.buffer['dw'][indices],
            "active": self.buffer['active'][indices]
        }

        return buffer, max_episode_len

        

