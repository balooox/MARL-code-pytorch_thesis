from smac.env import StarCraft2Env
from qmix_smac import QMIX_SMAC
from replay_buffer import ReplayBuffer

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

class QMIX_SMAC_Runner():
    def __init__(self, args, env_name) -> None:
        self.args = args
        self.env_name = '3m'
        self.number = 4
        self.seed = 0
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = StarCraft2Env(map_name=env_name)
        self.env.info = self.env.get_env_info()
        self.args.state_shape = self.env.info['state_shape'] # just a number, for current env: 48
        self.args.obs_shape = self.env.info['obs_shape'] # just a number, for current env: 30 
        self.args.n_actions = self.env.info['n_actions'] # just a number, for current env: 9
        self.args.n_agents = self.env.info['n_agents'] # just a number, for current env: 3
        self.args.episode_limit = self.env.info['episode_limit'] # just a number, for current env: 60
        self.played_episodes = 0
        self.num_steps = 0
        #print(self.env.info)
        
        self.epsilon = self.args.epsilon_start

        self.replay_buffer = ReplayBuffer(args)

        self.writer = SummaryWriter(log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}'.format('QMIX', 'QMIX', self.env_name, self.number, self.seed))

        self.agent_n = QMIX_SMAC(args)

    def run(self):
        
        n_eval = -1
        while self.num_steps < args.max_time_steps:
            if self.num_steps // args.evaluation_freq > n_eval:
                self.evaluate_policy()
                n_eval += 1

            _, _, episode_steps = self.run_episode_smac()
            self.num_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                self.agent_n.train(self.replay_buffer)
            

        self.env.close()


    def run_episode_smac(self, evaluate=False):
        
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.agent_n.eval_q_net.rnn_hidden = None

        last_onehot_a_n = np.zeros((self.args.n_agents, self.args.n_actions))

        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs() # (n_agents, obs_shape) -> (3, 30)
            s = self.env.get_state() #  (observation_features, ) -> (48, )
            avail_a_n = self.env.get_avail_actions() # (n_agents, n_actions) -> (3, 9)
            #print('avail_action')
            #print(avail_a_n)

            epsilon = 0 if evaluate else self.epsilon

            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            #print(a_n)
            #print('\n')
            
            # Note: We only receive a reward for the whole transition (not for every agent)
            r, done, info = self.env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            last_onehot_a_n = np.eye(self.args.n_actions)[a_n]
            episode_reward += r 
            
            if not evaluate:

                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True # we are done, because the game is finished (we are either dead or won) -> there is no next state
                else:
                    dw = False # we are done, because the epsisode limit was reached -> there would be a next state

                # Store the current transition
                # obs_n, s, avail_a_n, a_n, r, dw are stored in the transition item of this state
                # last_one_hot_a_n is stored in the transition item of the next state
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)

                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_item(episode_step + 1, obs_n, s, avail_a_n)

            self.played_episodes += 1

        return win_tag, episode_reward, episode_step

    def evaluate_policy(self):
        
        n_wins = 0
        eval_reward = 0
        for _ in range(self.args.evaluation_length):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                n_wins += 1
            eval_reward += episode_reward

        win_rate = n_wins / self.args.evaluation_length
        eval_reward = eval_reward / self.args.evaluation_length

        print(f"total_steps: {self.num_steps}, win rate: {win_rate}, evaluation reward: {eval_reward}")
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.num_steps) # add to tensorboard logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_steps', type=int, default=int(1e6))
    parser.add_argument('--evaluation_freq', type=int, default=5000)
    parser.add_argument('--evaluation_length', type=int, default=32)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epsilon_start', type=float, default=0.9)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_steps', type=int, default=50000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_freq', type=int, default=200)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--qmix_hidden_dim', type=int, default=32)
    parser.add_argument('--use_grad_clip', type=bool, default=True)

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon_start - args.epsilon_min) / args.epsilon_decay_steps
    env_name = '3m'

    runner = QMIX_SMAC_Runner(args, env_name=env_name)
    runner.run()

    