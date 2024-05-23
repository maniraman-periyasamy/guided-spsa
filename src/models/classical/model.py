import torch

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_layer,n_hidden_units, n_output):
        super(Net, self).__init__()
        
        layers = [torch.nn.Linear(n_feature, n_hidden_units)]
        layers.append(torch.nn.ReLU())
        for i in range(0, n_hidden_layer-1):
            layers.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
        self.predict = torch.nn.Linear(n_hidden_units, n_output)
        
    def forward(self, x):

        x = self.layers(x)
        x = self.predict(x)             # linear output
        return x