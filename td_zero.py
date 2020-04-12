from torch import tensor, max, device, float, inverse, sub, sqrt, mm, abs, exp


class TD_zero():
    def __init__(self, device: device, gamma=0.9):
        self.__gamma: float = gamma
        self.__valuePrevious: tensor = None
        self.__device = device

    def getTD(self, reinforcement: tensor, value_est: tensor, TDTargetOnly=True):
        # if self.__valuePrevious is None:
        #     result = tensor([[0]], device=self.__device, dtype=float)
        # else:
        #     # Выбираем, вычисляется ли просто TDTarget
        #     previousValue = 0 if TDTargetOnly else self.__valuePrevious
        #     result = reinforcement + self.__gamma * self.__valuePrevious - previousValue

        previousValue = 0 if TDTargetOnly else self.__valuePrevious
        result = reinforcement + self.__gamma * value_est - previousValue

        self.__valuePrevious = value_est

        # возвращаем результат, отсекая расчёт временных разниц от дерева вычислений (чтобы через него не было
        # обратного прохода
        return result.detach()

    def getPreviousValue(self):
        return self.__valuePrevious

    def setPreviousValue(self, previousValue: tensor):
        self.__valuePrevious = previousValue


class Reinforcement():
    def __init__(self):
        self.__reinforcementByClass = [1., 1., 1., 1., 1.]
        self.__punishmentByClass = [-1., -1., -1., -1., -1.]

    def getReinforcement(self, outputs: tensor, targets: tensor):
        (_, maxTargetIndex) = max(targets, 1)
        (_, maxOutputIndex) = max(outputs, 1)
        if maxTargetIndex.item() == maxOutputIndex.item():
            return tensor([[self.__reinforcementByClass[maxTargetIndex]]], dtype=float, device=outputs.device)
        else:
            # return tensor([[0]], dtype=float, device=outputs.device)
            return tensor([[self.__punishmentByClass[maxTargetIndex]]], dtype=float, device=outputs.device)


class AnyLoss():
    @classmethod
    def Qmaximization(cls, input: tensor, target: tensor):
        return mm(sub(target, input), sub(target, input)) + inverse(mm(target, input))

    @classmethod
    def Qinverse(cls, input: tensor, target: tensor):
        return mm(sub(target, input), sub(target, input)) + inverse(exp(input * 0.1))