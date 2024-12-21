from collections import defaultdict, deque
class Buffer():
    def __init__(self, observation_shape, action_shape, capacity=100000):
        self.storage = []
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.data = deque(maxlen=capacity)
    
    def add(self, data_samples):
        self.data.append(data_samples)
    
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)
    
    def __len__(self):
        return len(self.data)
    
    def reset(self):
        self.data.clear()
    
    def get_all_data(self):
        return self.data