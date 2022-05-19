import numpy as np

class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf

def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes

def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)
            
def update(node: Node, new_value: float):
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)
    
def propagate_changes(change: float, node: Node):
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)

class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size):

        self.alpha = 0.6
        self.beta = 0.4
        self.beta_steps = 180000
        self.beta_inc = (1 - self.beta) / self.beta_steps
        self.eps = 0.01
        
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
        priorities = np.zeros((self.mem_size, ), dtype=np.float32)

        self.root_node, self.leaf_nodes = create_tree(priorities)
        self.max_priority = -1.0
        self.absolute_error_upper = 1.0

    def store_transition(self, state, action, reward, state_, terminal):
        self.beta = min(self.beta + self.beta_inc,1)
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        update(self.leaf_nodes[index], abs(self.max_priority))

        self.mem_cntr += 1

    def sample_memory(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        """

        if self.mem_cntr > self.mem_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.mem_cntr]"""
        tree_total = self.root_node.value
        indices = []
        probs = []
        for i in range(self.batch_size):
            rand_val = np.random.uniform(0, tree_total)

            leaf = retrieve(rand_val, self.root_node)
            
            indices.append(leaf.idx)

            ###remove here IS
            #probs.append(leaf.value / tree_total)

        #and here IS
        #probs = np.array(probs,dtype=np.float32)

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        new_states = self.new_state_memory[indices]
        terminals = self.terminal_memory[indices]

        #Both of these IS
        #weights = (max_mem * probs) ** (-self.beta)
        #weights /= weights.max()

        #last bit here
        return states, actions, rewards, new_states, terminals, indices#, np.array(weights, dtype=np.float32)

    def batch_update(self,batch_indices, batch_priorities):
        #print(batch_priorities.type)

        #batch_priorities += self.eps
        #batch_priorities = np.minimum(batch_priorities, self.absolute_error_upper)

        #batch_priorities = np.power(batch_priorities, self.alpha)

        self.max_priority = max(self.max_priority, max(batch_priorities))
        
        for idx, prio in zip(batch_indices, batch_priorities):            
            update(self.leaf_nodes[idx], prio)

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size    
        

################################################ the below implementation doesnt use tree
class ReplayMemoryBuffer:
    def __init__(self, input_dims, max_mem, batch_size):

        self.alpha = 0.6
        self.beta = 0.4
        self.beta_steps = 50000
        self.beta_inc = (1 - self.beta) / self.beta_steps
        self.eps = 1e-5
        
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
        self.priorities = np.zeros((self.mem_size, ), dtype=np.float32)


    def store_transition(self, state, action, reward, state_, terminal):
        self.beta = min(self.beta + self.beta_inc,1)
        
        max_prio = self.priorities.max() if self.mem_cntr > 0 else 1.0
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.priorities[index] = max_prio

        self.mem_cntr += 1

    def sample_memory(self):

        max_mem = min(self.mem_cntr, self.mem_size)

        if self.mem_cntr > self.mem_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.mem_cntr]

        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(max_mem, self.batch_size, p=probs)

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        new_states = self.new_state_memory[indices]
        terminals = self.terminal_memory[indices]

        weights = (max_mem * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return states, actions, rewards, new_states, terminals, indices, np.array(weights, dtype=np.float32)

    def batch_update(self,batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size


    
