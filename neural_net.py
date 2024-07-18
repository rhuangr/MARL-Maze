import torch
import torch.nn as nn
from torch.optim import Adam
torch.manual_seed(2)
# represents the names of individual 
FEATURE_NAMES = ['direction', 'dead ends','visible marked cell', 'visible unmarked cell',
                  'visible_end', 'on marked cell', 'timestep', 'relative x', 'relative y']
# represents the dimensions of the feature vectors
FEATURE_DIMS = [2, 4, 4, 4, 4, 1, 1, 1, 1]
EMBEDDING_DIM = 10

class simple_nn(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU):
        super(simple_nn, self).__init__()
        self.projection = Projection()
        self.attention = Attention()
        self.layers = nn.ModuleList()
        self.activation = activation

        layer_sizes[0] = layer_sizes[0] * EMBEDDING_DIM
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # self.activations.append(activation)
        self.optimizer = Adam(self.parameters(), lr = 0.0001)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        attention_scores = self.attention(x)
        self.projection(x, attention_scores)
        x = self.projection(x)
        for i in range(len(self.layers) - 1):
            x = self.activation()(self.layers[i](x))
        x = self.layers[-1](x)
        return x, attention_scores


class Attention(nn.Module):

    def __init__(self, input_size=len(FEATURE_DIMS), middle_size=64, activation=nn.Sigmoid):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.attention = nn.Sequential(
            nn.Linear(input_size, middle_size),
            activation(),
            nn.Linear(middle_size, input_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        attention_scores = self.attention(input)
        return attention_scores
    
# transforms individual features into embeddings of equal size, then passed into attention layer
class Projection(nn.Module):

    def __init__(self, feature_dims=FEATURE_DIMS, embedding_dim=EMBEDDING_DIM, activation=nn.ReLU):
        super(Projection, self).__init__()
        print(feature_dims)
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(dim, embedding_dim) for dim in feature_dims])

    def forward(self, input, attention_scores):
        index = 0
        observations = []

        for i in range(len(self.feature_dims)):
            # print(input[:, index:index+self.feature_dims[i]])
            embedding = self.layers[i](input[:, index:index+self.feature_dims[i]])
            embedding = self.activation()(embedding)
            embedding = embedding * attention_scores[:, i].unsqueeze(1)
            index += self.feature_dims[i]
            observations.append(embedding)

        return torch.cat(observations, dim=1)
    
if __name__ == "__main__":
    x = torch.as_tensor(([1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0, 0, 0],
                         [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0, 0, 0]), dtype=torch.float32)
    y = torch.as_tensor([[2,3],[3,4],[4,5]], dtype=torch.float32)
    z = torch.as_tensor(([1,2,3,4,2,3,4,2,2],[1,2,3,4,2,3,4,2,2]), dtype=torch.float32)
    # z = []

    agent = simple_nn([22,4,5,])
    agent(x)