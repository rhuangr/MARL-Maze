import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

torch.manual_seed(3)
# represents the dimensions of the feature vectors, used for dynamic network creation
FEATURE_DIMS = [4,4,4,4,4,4,4,4,4,4,4,4,2,2,1,4,1,1,1,1,1,1,2]
FEATURE_AMOUNT = len(FEATURE_DIMS)
OBS_SPACE = np.sum(FEATURE_DIMS)
EMBEDDING_DIM = 10

class Actor(nn.Module):
    # note: layer size does not include first layer since it is static
    def __init__(self, hidden_sizes=[164,164,164,164,164], activation=nn.ReLU):
        super(Actor, self).__init__()
        self.projection = Projection()
        self.attention = m_Attention()
        self.layers = nn.ModuleList()
        self.activation = activation
        
        # dynamic layer creation
        self.layers.append(nn.Linear(FEATURE_AMOUNT * EMBEDDING_DIM, hidden_sizes[0]))     
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        self.move_head = nn.Linear(hidden_sizes[-1],5)
        self.mark_head = nn.Linear(hidden_sizes[-1],1)
        # self.signal_head = nn.Linear(hidden_sizes[-1], 1)
        self.initialize_weights()
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, OBS_SPACE)
        x = self.projection(x)
        x = self.attention(x)
        # x = torch.reshape(x,(-1,FEATURE_AMOUNT * EMBEDDING_DIM))
        for i in range(len(self.layers)-1):
            x = self.activation()(self.layers[i](x))
            
        x = self.activation()(self.layers[-1](x))
        move = self.move_head(x)
        mark = self.mark_head(x)
        # signal = self.signal_head(x)
        return [move,mark]
    
    def initialize_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
        with torch.no_grad():  
            self.move_head.weight *= 0.01
            self.mark_head.weight *= 0.01
            # self.signal_head.weight *= 0.1

# transforms individual features into embeddings of equal size, then passed into attention layer
class Projection(nn.Module):

    def __init__(self):
        super(Projection, self).__init__()
        self.layers = nn.ModuleList()
        for dim in FEATURE_DIMS:
            self.layers.append(nn.Linear(dim, EMBEDDING_DIM))

    def forward(self, input):
        index = 0
        observations = []
        for i in range(len(FEATURE_DIMS)):
            input_slice = input[:, index:index+FEATURE_DIMS[i]]
            embedding = self.layers[i](input_slice)
            observations.append(embedding)
        # print(torch.cat(observations,dim=1).reshape(-1, FEATURE_AMOUNT, EMBEDDING_DIM).shape)
        return torch.cat(observations,dim=1).reshape(-1, FEATURE_AMOUNT, EMBEDDING_DIM)

class m_Attention(nn.Module):
    def __init__(self, kq_dim=10):
        super(m_Attention, self).__init__()
        self.kq_dim = kq_dim
        self.keys = nn.Linear(EMBEDDING_DIM, kq_dim, bias=False)
        self.querys = nn.Linear(EMBEDDING_DIM, kq_dim, bias=False)
        self.values = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)

    def forward(self, input):
        keys = self.keys(input)
        querys = self.querys(input)
        values = self.values(input)
        logits = torch.einsum("bij,bkj->bik",querys,keys)/np.sqrt(self.kq_dim)
        omega = torch.softmax(logits, dim=-1)
        context = torch.einsum("bij,bjk->bik",omega, values)
        return (input+context).reshape(-1,FEATURE_AMOUNT*EMBEDDING_DIM)

class Critic(nn.Module):
    def __init__(self, agent_amount, hidden_sizes = [128,128], activation = nn.ReLU):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.agent_amount = agent_amount
        self.layers.append(nn.Linear(self.agent_amount*OBS_SPACE, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.initialize_weights()
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = torch.reshape(x, (-1, self.agent_amount*OBS_SPACE))
        for i in range(len(self.layers)-1):
            x = self.activation()(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    
    def initialize_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
        
if __name__ == "__main__":
    x = torch.as_tensor(([[1.,1,1,1, 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.]]), dtype=torch.float32)
    y = torch.as_tensor([[[2,3],[3,4],[4,5]],[[1,2],[2,3],[3,4]]], dtype=torch.float32)
    z = torch.as_tensor(([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]), dtype=torch.float32)
    
    
    a = torch.as_tensor([[0.2,0.3,0.5],[0.2,0.3,0.5]], dtype=torch.float32)
    b = torch.as_tensor([[0.3, 0.7]], dtype=torch.float32)
    # print(torch.einsum("ij,ik->ikj",a,b))
    x = [True]
