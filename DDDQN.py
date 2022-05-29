import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy,copy
import time
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable

"""
some rainbow parameters

-actions repeated 4 times (this might be dicey for sync issues) maybe can get away with this tho
-only learns once every 4 frames (this might be dicey for sync issues)
-batch size is 32
-try getting rewards -1 < r < 1

"""

class NoisyLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(T.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", T.zeros(1, in_features))
        self.register_buffer("epsilon_output", T.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(T.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        T.randn(self.epsilon_input.size(), out=self.epsilon_input)
        T.randn(self.epsilon_output.size(), out=self.epsilon_output)

        func = lambda x: T.sign(x) * T.sqrt(T.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * Variable(eps_out.t())
        noise_v = Variable(T.mul(eps_in, eps_out))
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class DuelingDeepQNetworkConv(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir,noisy=False,atoms = 51,Vmin=-10,Vmax=10,batch_size=32):
        super(DuelingDeepQNetworkConv, self).__init__()
        
        self.start = time.time()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        print("device: " + str(self.device))
        
        #if framestack was turned off, this needs to change
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4,padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2,padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3,stride = 1)

        if not noisy:
            self.fcA = nn.Linear(64*6*2, 512)
            self.fcV = nn.Linear(64*6*2, 512)
            self.V = nn.Linear(512, 1 * atoms)
            self.A = nn.Linear(512, n_actions * atoms)
        else:
            self.fcA = NoisyLinear(64*10*4, 512)
            self.fcV = NoisyLinear(64*10*4, 512)
            self.V = NoisyLinear(512, 1 * atoms)
            self.A = NoisyLinear(512, n_actions * atoms)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.batch_size = batch_size
        self.atoms = atoms

        delta_z = (Vmax - Vmin) / (atoms - 1)
        self.softmax = nn.Softmax(dim=1)
        self.supports = T.arange(Vmin,Vmax + delta_z,delta_z).to(self.device)

        self.to(self.device)

    def forward(self,state):
        input_size = len(state)
        observation = state.view(-1, 4, 96, 52)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))

        observation = observation.view(-1, 64*10*4)
        
        flatA = F.relu(self.fcA(observation))#problem is here
        flatV = F.relu(self.fcV(observation))
        
        V = self.V(flatV).view(input_size, 1, self.atoms)
        A = self.A(flatA).view(-1, self.n_actions, self.atoms)
        adv_mean = A.mean(dim=1, keepdim=True)

        return V + (A - adv_mean)

    def both(self, x):
        cat_out = self(x)#this means categorical out
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t)

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), "current_model" + str(int(time.time() - self.start)))

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load("current_model7043"))

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir,noisy=False,atoms = 51,Vmin=-10,Vmax=10,batch_size=32):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = T.device('cpu')#'cuda:0' if T.cuda.is_available() else
        self.n_actions = n_actions
        print("device: " + str(self.device))

        if not noisy:
            self.fc1 = nn.Linear(*input_dims, 512)
            self.V = nn.Linear(512, 1 * atoms)
            self.A = nn.Linear(512, n_actions * atoms)
        else:
            self.fc1 = NoisyLinear(*input_dims, 512)
            self.V = NoisyLinear(512, 1 * atoms)
            self.A = NoisyLinear(512, n_actions * atoms)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.batch_size = batch_size
        self.atoms = atoms

        delta_z = (Vmax - Vmin) / (atoms - 1)
        self.softmax = nn.Softmax(dim=1)
        self.supports = T.arange(Vmin,Vmax + delta_z,delta_z)

        self.to(self.device)
    
    #need to check many of these. Dimensions may be messed up
    def forward(self,state):
        input_size = len(state)
        flat = F.relu(self.fc1(state))
        V = self.V(flat).view(input_size, 1, self.atoms)
        A = self.A(flat).view(-1, self.n_actions, self.atoms)
        adv_mean = A.mean(dim=1, keepdim=True)

        return V + (A - adv_mean)

    def both(self, x):
        cat_out = self(x)#this means categorical out
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t)

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load("current_model7043"))

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims,batch_size,n_actions,
                 max_mem_size = 1000000, eps_end=0.01, eps_dec=2e-6, memory = "ER",
                 replace=100,image = False,framestack = True,learning_starts=10000,
                 preprocess = True,n_step = False,noisy=False,action_repeat=1):

        self.temp_timer = time.time()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.chkpt_dir = 'tmp/dueling_ddqn'
        self.memory_type = memory
        #self.framestack = framestack
        self.image = image
        self.stacked_frames = None
        self.need_replace = False
        self.before_learn = True
        self.before_learn_change = False
        self.learning_starts = learning_starts
        self.preprocess = preprocess

        self.action_repeat = action_repeat - 1
        self.cur_action_repeat = self.action_repeat + 1
        self.held_action = None

        #n_step parameters
        self.n_step = n_step
        self.held_rewards = []
        self.held_states = []
        self.held_actions = []

        self.gamma_n = self.gamma ** self.n_step

        #noisy
        self.noisy = noisy
        if self.noisy:
            self.epsilon = 0
            self.eps_min = 0

        #categorical params
        self.Vmax = 50
        self.Vmin = -1
        self.N_atoms = 51

        #increment in each atom
        self.delta_z = (self.Vmax - self.Vmin) / (self.N_atoms - 1)

        if memory == "ER":
            from ER import ReplayMemory
        elif memory == "PER":
            from PER import ReplayMemory
        else:
            raise Exception("Invalid Memory Algorithm")

        self.memory = ReplayMemory(input_dims,max_mem_size,self.batch_size)

        if not self.image:
            self.device = T.device('cpu')

            self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_eval',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy,atoms = self.N_atoms,
                                        Vmin=self.Vmin,Vmax=self.Vmax,batch_size=self.batch_size)

            self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_next',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy,atoms = self.N_atoms,
                                        Vmin=self.Vmin,Vmax=self.Vmax,batch_size=self.batch_size)
        else:
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            self.q_eval = DuelingDeepQNetworkConv(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_eval',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy,
                                        Vmin=self.Vmin,Vmax=self.Vmax,batch_size=self.batch_size)

            self.q_next = DuelingDeepQNetworkConv(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_next',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy,
                                        Vmin=self.Vmin,Vmax=self.Vmax,batch_size=self.batch_size)

    def distr_projection(self,next_distr, rewards, dones):
        """
        Perform distribution projection aka Catergorical Algorithm from the
        "A Distributional Perspective on RL" paper
        """
        proj_distr = np.zeros((self.batch_size, self.N_atoms), dtype=np.float32)
        for atom in range(self.N_atoms):
            tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards + (self.Vmin + atom * self.delta_z) * self.gamma_n))
            b_j = (tz_j - self.Vmin) / self.delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
        if dones.any():
            proj_distr[dones] = 0.0
            tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[dones]))
            b_j = (tz_j - self.Vmin) / self.delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
        return proj_distr
        

    def choose_action(self, observation):

        if self.cur_action_repeat < self.action_repeat:
            self.cur_action_repeat += 1
            return self.held_action
        else:
            self.cur_action_repeat = 0

        if self.memory.mem_cntr < self.learning_starts:
            self.held_action = np.random.choice(self.action_space)
            return self.held_action
        
        if self.image:
                        
            if np.random.random() > self.epsilon:
                state = T.tensor(np.array([observation],dtype=np.float32),dtype=T.float32).to(self.q_eval.device)
                q_vals = self.q_eval.qvals(state)
                action = T.argmax(q_vals).item()
            else:
                action = np.random.choice(self.action_space)

        else:
            if np.random.random() > self.epsilon:
                state = T.tensor([observation],dtype=T.float32).to(self.q_eval.device)
                q_vals = self.q_eval.qvals(state)
                action = T.argmax(q_vals).item()
            else:
                action = np.random.choice(self.action_space)            

        self.held_action = action
        return action

    def store_transition(self, state, action, reward, state_, done):
        if not self.n_step:
            if self.image:
                self.memory.store_transition(state, action, reward, state_, done)
                    
            else:
                self.memory.store_transition(state, action, reward, state_, done)

        else:
            if self.image:

                self.calc_n_step(state, action, reward,state_, done)
                
                #self.memory.store_transition(state, action, reward, self.stacked_frames, done)
                    
            else:
                self.calc_n_step(state, action, reward, state_, done)
                #self.memory.store_transition(state, action, reward, state_, done)

        if done:
            self.act_replace_target_network()
            if self.before_learn_change and self.before_learn:
                self.before_learn = False

            #choose new action at start of next episode
            self.cur_action_repeat = self.action_repeat + 1

    def calc_n_step(self,state, action, reward, state_, done):

        #update current memories
        for i in range(len(self.held_rewards)):
            self.held_rewards[i] += reward * (self.gamma ** (len(self.held_rewards)  - i))

        #add new memory
        self.held_rewards.append(reward)
        self.held_states.append(state)
        self.held_actions.append(action)

        #deal with finished memories
        if len(self.held_rewards) > self.n_step:
            n_state = deepcopy(self.held_states[0])
            n_reward = copy(self.held_rewards[0])
            n_action = copy(self.held_actions[0])

            self.memory.store_transition(n_state, n_action, n_reward, state_, done)
            del self.held_states[0]
            del self.held_rewards[0]
            del self.held_actions[0]

        #terminal states
        if done:
            x = copy(len(self.held_rewards))
            for i in range(x):
                n_state = deepcopy(self.held_states[0])
                n_reward = copy(self.held_rewards[0])
                n_action = copy(self.held_actions[0])

                self.memory.store_transition(n_state, n_action, n_reward, state_, done)
                del self.held_states[0]
                del self.held_rewards[0]
                del self.held_actions[0]     

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.need_replace = True

    def act_replace_target_network(self):
        if self.need_replace:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            self.need_replace = False

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        
        if self.memory.mem_cntr % 256 == 255:
            print("Frames per hour: " + str(3600 / ((time.time() - self.temp_timer)/256)))
            self.temp_timer = time.time()
        
        if self.memory.mem_cntr < self.learning_starts:
            return
        elif self.before_learn:
            self.before_learn_change = True
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        if self.memory_type == "PER":
            #remove tree_weights for no IS
            #states,actions,rewards,new_states,dones,tree_indices,tree_weights = self.memory.sample_memory()
            states,actions,rewards,new_states,dones,tree_indices = self.memory.sample_memory()
        else:
            states,actions,rewards,new_states,dones = self.memory.sample_memory()

        states = T.tensor(states).to(self.q_eval.device)
        #rewards = T.tensor(rewards).to(self.q_eval.device)
        #dones = T.tensor(dones).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        states_ = T.tensor(new_states).to(self.q_eval.device)

        distr_v, qvals_v = self.q_eval.both(T.cat((states, states_)))
        next_qvals_v = qvals_v[self.batch_size:]
        distr_v = distr_v[:self.batch_size]

        next_actions_v = next_qvals_v.max(1)[1]
        next_distr_v = self.q_next(states_)
        next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
        next_best_distr_v = self.q_next.apply_softmax(next_best_distr_v)
        next_best_distr = next_best_distr_v.data.cpu().numpy()

        dones = dones.astype(np.bool)

        proj_distr = self.distr_projection(next_best_distr, rewards, dones)

        state_action_values = distr_v[range(self.batch_size), actions]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)
        proj_distr_v = T.tensor(proj_distr).to(self.device)

        loss_v = -state_log_sm_v * proj_distr_v
        loss_array = loss_v.sum(dim=1)

        loss = loss_array.mean()
        
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        if self.memory_type == "PER":
            #add epsilon
            
            loss_array += self.memory.eps
            loss_array = T.clamp(loss_array,min=0.0, max=1.0)
            loss_array = T.pow(loss_array,self.memory.alpha)

            #update tree priorities
            self.memory.batch_update(tree_indices, loss_array.data.cpu().numpy())

