import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import time


class DuelingDeepQNetworkConv(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir):
        super(DuelingDeepQNetworkConv, self).__init__()
        
        self.start = time.time()
        #if framestack was turned off, this needs to change
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4,padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2,padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3,stride = 1)
        self.fc1 = nn.Linear(64*6*2, 512)

        self.A = nn.Linear(512, n_actions)
        self.V = nn.Linear(512, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("device: " + str(self.device))
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
        T.save(self.state_dict(), "current_model" + str(round(time.time() - self.start,3)))

    def load_checkpoint(self):
        #print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims,chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

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
                 preprocess = True):

        #need to test image
        #need to add preprocess

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = 6.25e-5#lr
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
                                       chkpt_dir=self.chkpt_dir)

            self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_next',
                                       chkpt_dir=self.chkpt_dir)
        else:
            #need to implement framestack
            self.q_eval = DuelingDeepQNetworkConv(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_eval',
                                       chkpt_dir=self.chkpt_dir)

            self.q_next = DuelingDeepQNetworkConv(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name='lunar_lander_dueling_ddqn_q_next',
                                       chkpt_dir=self.chkpt_dir)

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
        state_ = self.process_frame(state_)
        state = deepcopy(self.stacked_frames)
        self.stack_frames(state_)
        self.memory.store_transition(state, action, reward, self.stacked_frames, done)
        if done:
            self.stacked_frames = None
            self.act_replace_target_network()
            if self.before_learn_change and self.before_learn:
                self.before_learn = False

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
        if self.memory.mem_cntr < self.learning_starts:
            return
        elif self.before_learn:
            self.before_learn_change = True
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        if self.memory_type == "PER":
            tree_idx, states,actions,rewards,new_states,dones = self.memory.sample_memory()
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
            loss_array = T.abs(q_target - q_pred).to(self.q_eval.device)#T.absolute(q_target - q_pred).to(self.q_eval.device)
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)  
        
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        if self.memory_type == "PER":
            #update tree priorities
            self.memory.batch_update(tree_idx, loss_array.cpu().detach())

