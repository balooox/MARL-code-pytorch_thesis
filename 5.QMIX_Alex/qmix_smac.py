import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mix_net import QMIX_Net

class Q_Network_RNN(nn.Module):
    def __init__(self, args, input_shape):
        super(Q_Network_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape = (agent_n, input_dim)
        # When 'train', inputs.shape = (batch_size*agent_n, input_dim)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q
    


class QMIX_SMAC():
    def __init__(self, args) -> None:
        self.args = args

        # Calculate input features
        # (1, Anzahl Features Observation + Anzahl Features der Aktion + Anzahl der Features für Agent)
        # (1, 30 + 9 + 3) = (1, 42)
        self.input_shape = self.args.obs_shape + self.args.n_actions + self.args.n_agents
        # print(self.input_shape)

        # agent network
        self.eval_q_net = Q_Network_RNN(args, self.input_shape)
        self.target_q_net = Q_Network_RNN(args, self.input_shape)
        self.target_q_net.load_state_dict(self.eval_q_net.state_dict())

        # mixing network
        self.eval_mix_net = QMIX_Net()
        self.target_mix_net = QMIX_Net()
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        
        if np.random.rand() < epsilon:
            # print('random_action')
            return [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
        else:
            # print('best_action')
            inputs = np.concatenate((obs_n, last_onehot_a_n), axis=1) # add last_onehot_a_n 
            inputs = np.concatenate((inputs, np.eye(3, dtype=float)), axis=1) # add agent id
            inputs = torch.from_numpy(inputs)

            q_value = self.eval_q_net(inputs.float())
            # print('q_value')
            # print(q_value)

            q_value = torch.where(torch.tensor(avail_a_n, dtype=bool), q_value, torch.tensor(-999999.0))
            # print('avail_q_value')
            # print(q_value)


            return (torch.max(q_value, 1)[1]).detach().numpy()

    def train(self, replay_buffer):
        
        # get batch from replay buffer
        batch, max_episode_len = replay_buffer.sample()
        
        # reest the hidden state of the agent networks
        self.eval_q_net.rnn_hidden = None
        self.target_q_net.rnn_hidden = None

        q_evals, q_targets = [], []
        for t in range(1):
            ### Get q value ###
            
            cur_batch_size = 2

            obs_n = batch['obs_n'][:cur_batch_size,t]
            last_onehot_a_n = batch['last_onehot_a_n'][:cur_batch_size,t]
            a_n = batch['a_n'][:cur_batch_size, t]
            avail_a_n = batch['avail_a_n'][:cur_batch_size, t]

            ## Calculate input ##
            print('---obs_n---')
            print(obs_n)
            print('---last_onehot_a_n---')
            print(last_onehot_a_n)

            # Concate obs and action
            inputs = np.concatenate((obs_n, last_onehot_a_n), axis=2)
            print('---inputs---')
            print(inputs)

            # Create onehot encoded agent id
            agent_id_one_hot = np.stack([np.eye(self.args.n_agents)] * cur_batch_size)
            print('---agent_id_one_hot---')
            print(agent_id_one_hot)

            # Add agent id to inputs
            inputs = np.concatenate((inputs, agent_id_one_hot), axis=2)
            print('---inputs---')
            print(inputs)

            # Reshape inputs from (batch_size, n_agent, input_shape) to (batch_size * n_agent, input_shape)
            inputs = inputs.reshape(-1, self.input_shape)
            # Convert to tensor
            inputs = torch.from_numpy(inputs).float()

            # Get q values
            q_eval = self.eval_q_net(inputs)
            print(f'---q_value, t:{t}---')
            print(q_eval)

            ## Get the q value of the action the agent chose ##
            print('---a_n---')
            print(a_n)

            # Reshape a_n from (batch_size,n_agents) to (batch_size * n_agents,)
            a_n = a_n.reshape(cur_batch_size * self.args.n_agents, -1)
            print(a_n)

            # Get q value of chosen action 
            q_eval = torch.gather(q_eval, 1, torch.from_numpy(a_n).long())
            print('---q_value of chosen action---')
            print(q_eval)

            # Reshape q_value back from (batch_size*n_agent, input_shape) to (batch_size, n_agent, 1)
            q_eval = q_eval.reshape(cur_batch_size, self.args.n_agents, -1)
            print(q_eval)
            print(q_eval.shape)


            ### Get q target ###

            obs_n = batch['obs_n'][:cur_batch_size,t+1]
            last_onehot_a_n = batch['last_onehot_a_n'][:cur_batch_size,t+1]
            inputs = np.concatenate((obs_n, last_onehot_a_n), axis=2)
            inputs = np.concatenate((inputs, agent_id_one_hot), axis=2)
            inputs = inputs.reshape(-1, self.input_shape)
            inputs = torch.from_numpy(inputs).float()

            q_target = self.target_q_net(inputs)
            print(f'---q target, t:{t}')
            print(q_target)
            q_target = q_target.reshape(cur_batch_size, self.args.n_agents, -1)

            ## Get the max q target of each step

            print('---avail_a_n---')
            print(avail_a_n)

            q_target = torch.where(torch.tensor(avail_a_n, dtype=bool), q_target, torch.tensor(-9)) #TODO: -999
            print('---q_target---')
            print(q_target)

            q_target = torch.max(q_target, 2)[0]

            print('---q_target---')
            print(q_target)

            exit()

            ### Append the values to arrays ###

            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        print('---q_evals---')
        print(q_evals)
        print('---q_targets---')
        print(q_targets)



            


        #eval_q_values = self.get_cur_q_value(obs_n, last_onehot_a_n, a_n)
        


        


        exit()
    
    def get_cur_q_value(self, obs_n, last_onehot_a_n, a_n):
        

        print('---obs---')
        print(obs_n)
        print('---last_one_hot_a_n')
        print(last_onehot_a_n)

        inputs = np.concatenate((obs_n, last_onehot_a_n), axis=2)
        inputs = np.concatenate((inputs, np.stack([np.eye(3)] * 2)), axis=2)

        inputs = inputs.reshape(-1, 42)
        print('---inputs----')
        print(inputs)
        print(inputs.shape)
        inputs = torch.from_numpy(inputs).float()
        
        eval_q_values = self.eval_q_net(inputs)
        print('---q_value---')
        print(eval_q_values)
        print(eval_q_values.shape)
        print('---a_n---')
        print(a_n)
        a_n = np.concatenate(a_n, axis=None)
        print(a_n)
        row_indices = torch.arange(eval_q_values.size(0))
        eval_q_values = eval_q_values[row_indices, a_n]
        print('needed q values')
        print(eval_q_values)

        return eval_q_values

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])
        inputs.append(batch['last_onehot_a_n'])

        agent_id_one_hot = torch.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).repeat(self.args.batch_size, max_episode_len + 1, 1, 1)
        inputs.append(agent_id_one_hot)

        print(inputs)
        print(inputs.shape)

        # inputs = torch.cat([x for x in inputs], dim=-1)

        # return inputs

    
