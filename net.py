import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, initWaights=False):
        """

        :param initWeights: инициализировать веса или нет? Инициализацию надо производить ТОЛЬКО при создании NN
        """
        self.__inNeuron = nn.Sigmoid()
        self.__h0Linear = nn.Linear(10, 10, bias=False)
        self.__h0Neuron = nn.Sigmoid()
        self.__OutLinear = nn.Linear(10, 5, bias=False)
        self.__OutNeuron = nn.Sigmoid()

        if initWaights:
            self.__initializeWeights()

    def __initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = self.__InNeuron(x)
        x = self.__h0Linear(x)
        x = self.__h0Neuron(x)
        x = self.__OutLinear(x)
        x = self.__OutNeuron(x)