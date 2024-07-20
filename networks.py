import torch
import torch.nn as nn
from torch.optim import Adam

torch.manual_seed(3)
# represents the dimensions of the feature vectors
FEATURE_DIMS = [2, 4, 4, 4, 4, 1, 1, 1, 1, 1]
FEATURE_AMOUNT = len(FEATURE_DIMS)
OBS_VECTOR_SIZE = 23
EMBEDDING_DIM = 10

class simple_nn(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, attention_layers=2):
        super(simple_nn, self).__init__()
        self.projection = Projection()
        self.attention = Attention(attention_layers)
        self.layers = nn.ModuleList()
        self.activation = activation

        # dynamic layer creation
        self.layers.append(nn.Linear(FEATURE_AMOUNT * EMBEDDING_DIM, layer_sizes[0]))
        # self.layers.append(nn.LayerNorm(layer_sizes[0]))
        
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # self.layers.append(nn.LayerNorm(layer_sizes[i+1]))
            
        self.optimizer = Adam(self.parameters(), lr = 0.0001)
        
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 23)
        x = self.projection(x)
        x, attention_scores = self.attention(x)
        
        for i in range(len(self.layers) - 1):
            x = self.activation()(self.layers[i](x))
        x = self.layers[-1](x)
        
        # code for layer norm
        # for i in range((len(self.layers)//2)-1):
        #     x = self.layers[i*2](x)
        #     x = self.layers[i*2+1](x)
        #     x = self.activation()(x)
        # x = self.layers[-2](x)
        # x = self.activation()(x)

        return x, attention_scores

# transforms individual features into embeddings of equal size, then passed into attention layer
class Projection(nn.Module):

    def __init__(self,activation=nn.ReLU):
        super(Projection, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(dim, EMBEDDING_DIM) for dim in FEATURE_DIMS])

    def forward(self, input):
        index = 0
        observations = []
        for i in range(len(FEATURE_DIMS)):
            embedding = self.layers[i](input[:, index:index+FEATURE_DIMS[i]])
            embedding = self.activation()(embedding)
            observations.append(embedding)
        return torch.cat(observations, dim=1)

class Attention(nn.Module):

    def __init__(self, layers, middle_size=64, activation=nn.ReLU, temperature=2):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.layers = nn.ModuleList()
        self.activation = activation

        # dynamic layer creation
        self.layers.append(nn.Linear(FEATURE_AMOUNT*EMBEDDING_DIM, middle_size))
        for i in range(layers - 2):
            self.layers.append(nn.Linear(middle_size,middle_size))
        self.layers.append(nn.Linear(middle_size, FEATURE_AMOUNT))

    def forward(self, input):
        
        logits = input
        for i in range(len(self.layers)):
            logits = self.layers[i](logits)
            self.activation()(logits)
        attention_scores = nn.Softmax(dim=-1)(logits/self.temperature)

        input = input.reshape(-1, FEATURE_AMOUNT, EMBEDDING_DIM)
        weighted_features = torch.einsum("ijk,ij->ijk", input, 1+attention_scores).reshape(-1,FEATURE_AMOUNT*EMBEDDING_DIM)
        return weighted_features, attention_scores
    
if __name__ == "__main__":
    x = torch.as_tensor(([[1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.]]), dtype=torch.float32)
    y = torch.as_tensor([[2,3],[3,4],[4,5]], dtype=torch.float32)
    z = torch.as_tensor(([[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]), dtype=torch.float32)
    
    x = nn.Linear(22,22)(x)
    print(x)
    x = nn.LayerNorm(22)(x)
    print(x)