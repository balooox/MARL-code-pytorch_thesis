from smac.env import StarCraft2Env
from qmix_smac import QMIX_SMAC
from replay_buffer import ReplayBuffer

import argparse
import tensorboard

import numpy as np

class QMIX_SMAC_Runner():
    def __init__(self, args, env_name) -> None:
        self.args = args
        self.env = StarCraft2Env(map_name=env_name)
        self.env.info = self.env.get_env_info()
        self.args.state_shape = self.env.info['state_shape']
        self.args.obs_shape = self.env.info['obs_shape'] # nur eine Zahl, 30
        self.args.n_actions = self.env.info['n_actions']
        self.args.n_agents = self.env.info['n_agents']
        self.args.episode_limit = self.env.info['episode_limit']
        # print(self.env.info)
        self.epsilon = self.args.epsilon_start

        self.replay_buffer = ReplayBuffer(args)

        self.writter = None # add tensorboard logger

        self.agent_n = QMIX_SMAC(args)

    def run(self):
        
        num_steps = 0
        while num_steps < args.max_time_steps:
            if num_steps % args.evaluation_freq == 0:
                self.evaluate_policy()

            _, _, episode_steps = self.run_episode_smac()
            num_steps += episode_steps

            if self.replay_buffer.current_size > self.args.batch_size:
                self.agent_n.train(self.replay_buffer)
            

        self.env.close()


    def run_episode_smac(self, evaluate=False):
        
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.agent_n.eval_q_net.rnn_hidden = None

        last_onehot_a_n = np.zeros((self.args.n_agents, self.args.n_actions))

        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs() # (n_agents, obs_shape) -> (3,30)
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            #print('avail_action')
            #print(avail_a_n)

            epsilon = 0 if evaluate else self.epsilon

            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            #print(a_n)
            #print('\n')
            
            r, done, info = self.env.step(a_n)
            win_tag = True if done and info['battle_won'] else False
            last_onehot_a_n = np.eye(self.args.n_actions)[a_n]
            episode_reward += r
            
            if not evaluate:

                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)

                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_item(episode_step, obs_n, s, avail_a_n)

        return win_tag, episode_reward, episode_step

        


    
    def evaluate_policy(self):
        
        n_wins = 0
        eval_episode_reward = []

        for evaluation_epiosde in range(self.args.evaluation_length):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            n_wins += win_tag
            eval_episode_reward.append(episode_reward)

        average_reward = sum(eval_episode_reward) / len(eval_episode_reward)
        perc_wins = n_wins / self.args.evaluation_length

        print(f"Wins: {perc_wins}, average reward: {average_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_steps', type=int, default=1e6)
    parser.add_argument('--evaluation_freq', type=int, default=5e4)
    parser.add_argument('--evaluation_length', type=int, default=50)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epsilon_start', type=float, default=0.9)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_steps', type=int, default=50000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_freq', type=int, default=10)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rnn_hidden_dim', type=int, default=64)

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon_start - args.epsilon_min) / args.epsilon_decay_steps
    env_name = '3m'

    runner = QMIX_SMAC_Runner(args, env_name=env_name)
    runner.run()

    