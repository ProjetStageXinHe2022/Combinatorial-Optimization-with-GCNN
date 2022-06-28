import pickle as pkl
import json
import ast

"""
param:path of folder
return :data_train_X,data_train_Y,data_validation_X,data_validation_Y,data_test_X,data_test_Y
"""


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.activate = nn.ReLU()

    def forward(self, x, A):
        x = self.linear(x)
        x = self.activate(x)
        x = A.mm(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channel, aux_channel, out_channel):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(aux_channel, aux_channel),
            nn.ReLU(),
            nn.Linear(aux_channel, aux_channel),
        )
        self.mlp = MLP(in_channel, aux_channel)
        self.outlayer = nn.Sequential(
            nn.Linear(aux_channel, aux_channel),
            nn.ReLU(),
            nn.Linear(aux_channel, out_channel),
            #             nn.LogSoftmax()
            nn.Sigmoid()
        )

    def forward(self, x, A, nb_net):
        x = self.mlp(x, A)
        for i in range(nb_net):
            x = self.layer(x)
        x = self.outlayer(x)
        return x





split = [0.70,0.15,0.15]
dataSet = load_data()
data_train,data_validation,data_test = torch.utils.data.random_split(dataSet,[int(len(dataSet) * s) for s in split])