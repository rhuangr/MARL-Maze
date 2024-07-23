import torch
import torch.nn as nn
from torch.optim import Adam
from numpy import sum,sqrt

torch.manual_seed(3)
# represents the dimensions of the feature vectors, used for dynamic network creation
FEATURE_DIMS = [4, 4, 4, 4, 4, 4, 4, 4, 1, 2]
FEATURE_AMOUNT = len(FEATURE_DIMS)
OBS_SPACE = sum(FEATURE_DIMS)
EMBEDDING_DIM = 40

class Brain(nn.Module):
    # note: layer size does not include first layer since it is static
    def __init__(self, actor=True, hidden_sizes=[164,164,164,164,164], activation=nn.ReLU):
        super(Brain, self).__init__()
        self.projection = Projection()
        self.attention = m_Attention()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.actor = actor
        # dynamic layer creation
        self.layers.append(nn.Linear(FEATURE_AMOUNT * EMBEDDING_DIM, hidden_sizes[0]))     
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        if actor:
            self.move_head = nn.Linear(hidden_sizes[-1],5)
            self.mark_head = nn.Linear(hidden_sizes[-1],1)
            self.signal_head = nn.Linear(hidden_sizes[-1], 1)
        else:
            self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.optimizer = Adam(self.parameters(), lr = 0.0001)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, OBS_SPACE)
        x = self.projection(x)
        x = self.attention(x)
        # x = torch.reshape(x,(-1,FEATURE_AMOUNT * EMBEDDING_DIM))
        for i in range(len(self.layers)-1):
            x = self.activation()(self.layers[i](x))
            
        if self.actor:
            x = self.activation()(self.layers[-1](x))
            move = self.move_head(x)
            mark = self.mark_head(x)
            signal = self.signal_head(x)
            return [move,mark,signal]
        else:
            x = self.layers[-1](x)
            return x
        # signal = self.signal_head(x)

# transforms individual features into embeddings of equal size, then passed into attention layer
class Projection(nn.Module):

    def __init__(self,activation=nn.ReLU):
        super(Projection, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for dim in FEATURE_DIMS:
            variable_size = EMBEDDING_DIM - dim if dim != 1 else EMBEDDING_DIM
            self.layers.append(nn.Linear(dim, variable_size))
            self.layers.append(nn.Linear(variable_size,EMBEDDING_DIM))
            self.layers.append(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM))

    def forward(self, input):
        index = 0
        observations = []
        for i in range(len(FEATURE_DIMS)):
            input_slice = input[:, index:index+FEATURE_DIMS[i]]
            embedding = self.layers[i*3](input_slice)
            embedding = self.layers[i*3+1](embedding)
            embedding = self.activation()(self.layers[i*3+2](embedding))
            observations.append(embedding)
        print(torch.cat(observations,dim=1).reshape(-1, FEATURE_AMOUNT, EMBEDDING_DIM).shape)
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
        omega = torch.softmax(torch.einsum("bij,bkj->bik",querys,keys)/sqrt(self.kq_dim), dim=-1)
        context = torch.einsum("bij,bjk->bik",omega, values)
        return (input+context).reshape(-1,FEATURE_AMOUNT*EMBEDDING_DIM)


if __name__ == "__main__":
    x = torch.as_tensor(([[1.,1,1,1, 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.]]), dtype=torch.float32)
    y = torch.as_tensor([[[2,3],[3,4],[4,5]],[[1,2],[2,3],[3,4]]], dtype=torch.float32)
    z = torch.as_tensor(([[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]), dtype=torch.float32)
    
    # test = nn.Linear(2,10)(y)
    # print(test)
    # test = Brain([25,25,4])
    # a, b = test(x)
    # print()
    # print()
    # print(a)
    # print(b)
    # print(test)
    # # print(torch.softmax(test,dim=-1))
    # x = torch.einsum("bij,bkj->bik",y,y)
    # print(torch.einsum("bij,bkj->bik",y,y))
    # print(torch.einsum("bij,bjk->bik",x,y))
    
    a = torch.as_tensor([[0.2,0.3,0.5],[0.2,0.3,0.5]], dtype=torch.float32)
    b = torch.as_tensor([[0.3, 0.7]], dtype=torch.float32)
    
    x = [1,2,3]
    x =~x
    print(x)
    # print(torch.einsum("ij,ik->ikj",a,b))