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
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir,noisy=False):
        super(DuelingDeepQNetworkConv, self).__init__()
        
        self.start = time.time()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("device: " + str(self.device))
        
        #if framestack was turned off, this needs to change
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4,padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2,padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3,stride = 1)


        if not noisy:
            self.fc1 = nn.Linear(64*6*2, 512)
            self.V = nn.Linear(512, 1)
            self.A = nn.Linear(512, n_actions)
        else:
            self.fc1 = NoisyLinear(64*6*2, 512)
            self.V = NoisyLinear(512, 1)
            self.A = NoisyLinear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.to(self.device)

    def forward(self, observation):
        
        #observation = T.Tensor(observation).to(self.device)
        observation = observation.view(-1, 4, 64, 32)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))

        observation = observation.view(-1, 64*6*2)
        observation = F.relu(self.fc1(observation))
        A = self.A(observation)
        V = self.V(observation)

        return V,A

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), "current_model" + str(int(time.time() - self.start)))

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir,noisy=False):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = T.device('cpu')#'cuda:0' if T.cuda.is_available() else 
        print("device: " + str(self.device))

        if not noisy:
            self.fc1 = nn.Linear(*input_dims, 512)
            self.V = nn.Linear(512, 1)
            self.A = nn.Linear(512, n_actions)
        else:
            self.fc1 = NoisyLinear(*input_dims, 512)
            self.V = NoisyLinear(512, 1)
            self.A = NoisyLinear(512, n_actions)        


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.to(self.device)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.V.reset_noise()
        self.A.reset_noise()

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        #print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims,batch_size,n_actions,
                 max_mem_size = 1000000, eps_end=0.01, eps_dec=2e-6, memory = "ER",
                 replace=100,image = False,framestack = True,learning_starts=10000,
                 preprocess = True,n_step = False,noisy=False):

        #need to test image
        #need to add preprocess
        #self.temp_timer = time.time()
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
        self.framestack = framestack
        self.image = image
        self.stacked_frames = None
        self.need_replace = False
        self.before_learn = True
        self.before_learn_change = False
        self.learning_starts = learning_starts
        self.preprocess = preprocess

        #n_step parameters
        self.n_step = n_step
        self.held_rewards = []
        self.held_states = []
        self.held_actions = []

        #noisy
        self.noisy = noisy
        if self.noisy:
            self.epsilon = 0
            self.eps_min = 0

        if memory == "ER":
            from ER import ReplayMemory
        elif memory == "PER":
            from PER import ReplayMemory
        else:
            raise Exception("Invalid Memory Algorithm")

        self.memory = ReplayMemory(input_dims,max_mem_size,self.batch_size)

        if not self.image:

            self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_eval',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy)

            self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_next',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy)
        else:
            #need to implement framestack
            self.q_eval = DuelingDeepQNetworkConv(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_eval',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy)

            self.q_next = DuelingDeepQNetworkConv(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_next',
                                       chkpt_dir=self.chkpt_dir,noisy = self.noisy)

    def stack_frames(self, frame, buffer_size=4):
        if self.stacked_frames is None:
            self.stacked_frames = np.zeros((buffer_size, *frame.shape))
            for idx, _ in enumerate(self.stacked_frames):
                self.stacked_frames[idx,:] = frame
        else:

            self.stacked_frames = np.roll(self.stacked_frames,-1)

            #3 is most recent frame
            self.stacked_frames[buffer_size-1,:] = frame
            #shape is (4,32,64)


    def choose_action(self, observation):
        
        if self.image:
            if self.stacked_frames is None:
                observation = self.process_frame(observation)
                self.stack_frames(observation)
                        
            if np.random.random() > self.epsilon:
                state = T.tensor(self.stacked_frames,dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)

        else:
            if np.random.random() > self.epsilon:
                state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)            

        return action

    def store_transition(self, state, action, reward, state_, done):
        if not self.n_step:
            if self.image:
                state_ = self.process_frame(state_)
                state = deepcopy(self.stacked_frames)
                self.stack_frames(state_)
                self.memory.store_transition(state, action, reward, self.stacked_frames, done)
                if done:
                    self.stacked_frames = None
                    
            else:
                self.memory.store_transition(state, action, reward, state_, done)

        else:
            if self.image:
                state_ = self.process_frame(state_)
                state = deepcopy(self.stacked_frames)
                self.stack_frames(state_)

                self.calc_n_step(state, action, reward, self.stacked_frames, done)
                
                #self.memory.store_transition(state, action, reward, self.stacked_frames, done)
                if done:
                    self.stacked_frames = None
                    
            else:
                self.calc_n_step(state, action, reward, state_, done)
                #self.memory.store_transition(state, action, reward, state_, done)

        if done:
            self.act_replace_target_network()
            if self.before_learn_change and self.before_learn:
                self.before_learn = False

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
                
    def process_frame(self,frame):
        #could try using half precision if needed
        frame = np.true_divide(frame, 255).astype(np.float32)
        return frame

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
        """
        if self.memory.mem_cntr % 32 == 31:
            print((time.time() - self.temp_timer)/32)
            self.temp_timer = time.time()
        """
        
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
        rewards = T.tensor(rewards).to(self.q_eval.device)
        dones = T.tensor(dones).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        states_ = T.tensor(new_states).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        #loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        if self.memory_type == "ER":
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        else:        

            #remove These for no IS
            #batch_weights_v = Variable(T.from_numpy(tree_weights))
            #loss_array = batch_weights_v * (q_pred - q_target) ** 2
            loss_array = (q_pred - q_target) ** 2
            loss = loss_array.mean()

            #loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        if self.memory_type == "PER":
            #add epsilon
            
            loss_array += self.memory.eps
            loss_array = T.clamp(loss_array,min=0.0, max=1.0)
            loss_array = T.float_power(loss_array,self.memory.alpha)

            #update tree priorities
            self.memory.batch_update(tree_indices, loss_array.data.cpu().numpy())#.cpu().detach()

