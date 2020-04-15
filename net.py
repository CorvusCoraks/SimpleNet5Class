import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, initWaights=False):
        """

        :param initWeights: инициализировать веса или нет? Инициализацию надо производить ТОЛЬКО при создании NN
        """
        super(ActorNet, self).__init__()
        
        self.__inNeuron = nn.Sigmoid()
        self.__h0Linear = nn.Linear(10, 10, bias=False)
        self.__h0Neuron = nn.Sigmoid()
        self.__outLinear = nn.Linear(10, 5, bias=False)
        self.__outNeuron = nn.Sigmoid()

        if initWaights:
            self.__initializeWeights()

    def __initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = self.__inNeuron(x)
        x = self.__h0Linear(x)
        x = self.__h0Neuron(x)
        x = self.__outLinear(x)
        x = self.__outNeuron(x)

        return x


class CriticNet(nn.Module):
    def __init__(self, initWaights=False):
        """

        :param initWeights: инициализировать веса или нет? Инициализацию надо производить ТОЛЬКО при создании NN
        """
        super(CriticNet, self).__init__()

        self.__inputFeatures = 10+5+1
        self.__outputFeatures = 1

        self.__inNeuron = nn.Sigmoid()
        self.__h0Linear = nn.Linear(self.__inputFeatures, self.__inputFeatures, bias=False)
        self.__h0Neuron = nn.Sigmoid()
        self.__h0Dropout = nn.Dropout(0.1)
        self.__h1Linear = nn.Linear(self.__inputFeatures, self.__inputFeatures, bias=False)
        self.__h1Neuron = nn.Sigmoid()
        self.__h1Dropout = nn.Dropout(0.1)
        # self.__h2Linear = nn.Linear(self.__inputFeatures, self.__inputFeatures, bias=False)
        # self.__h2Neuron = nn.Sigmoid()
        self.__outLinear = nn.Linear(self.__inputFeatures, self.__outputFeatures, bias=False)
        self.__outNeuron = nn.Sigmoid()

        if initWaights:
            self.__initializeWeights()

    def __initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = self.__inNeuron(x)
        x = self.__h0Linear(x)
        x = self.__h0Neuron(x)
        x = self.__h0Dropout(x)
        x = self.__h1Linear(x)
        x = self.__h1Neuron(x)
        x = self.__h1Dropout(x)
        # x = self.__h2Linear(x)
        # x = self.__h2Neuron(x)
        x = self.__outLinear(x)
        x = self.__outNeuron(x)

        return x