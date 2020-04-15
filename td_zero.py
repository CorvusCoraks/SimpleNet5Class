from torch import tensor, max, device, float, inverse, sub, sqrt, mm, abs, exp, div
import random


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
        self.__reinforcementByClass = [5., 5., 5., 5., 5.]
        self.__punishmentByClass = [-0.005, -0.005, -0.005, -0.005, -0.005]

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


class EnvironmentSearch():
    """
    Класс исследования окружающей среды.
    """
    # После нулевой эпохи определяется количество ошибочных действий актора по каждому классу.
    # Если в каком либо классе существенно мало или вобще 0 ошибочных действий, это значит,
    # что поданному классу не происходит исследования окружающей среды.
    #
    # Алгоритм исследования пространства состояний:
    #
    # На каждом действии актора с некоторой вероятностью на вход критика подётся исследовательское действие,
    # выбранное случайным образом из всего диапазона возможных действий в данном состоянии, плюс подкрепление за это
    # действие. Обратный проход в этом случае делается только по критику с целью его обучения.
    #
    # Чем точнее критик аппроксимирует функцию ценности, тем меньше производится исследовательских проходов через него.
    #
    # На основании вышеизложеннного. Эпохи чередуются: эпоха обучения без исследований, эпоха обучения с исследованиями
    # (на основании точности аппроксимации функции ценности из предыщущей эпохи)
    TD_cumul: tensor
    Qpr_cumul: tensor
    isInvestigation: bool = False
    investigationProbability = 0.

    @classmethod
    def probability(cls, batchCount):
        """
        Возвращает вероятность исследовательского прохода

        :param Qtarget:
        :param Q:
        :return:
        """
        deltaMedium = div(sub(EnvironmentSearch.TD_cumul, EnvironmentSearch.Qpr_cumul),batchCount)
        delta = sqrt(mm(deltaMedium, deltaMedium))
        if delta.item()*5 >= 0.95:
            return 0.95
        else:
            return delta

    @classmethod
    def setInvestigation(cls, investEpoch: bool):
        """
        Функция определяет, производится ли на данном проходе исследование
        :param TD_target:
        :param Qprevious:
        :return:
        """
        # Во входных данных вполне могут оказаться None
        if EnvironmentSearch.TD_cumul is None or EnvironmentSearch.Qpr_cumul is None:
            EnvironmentSearch.isInvestigation = False
            return

        if investEpoch:
            randomNumber = random.uniform(0, 1)
            # probability = EnvironmentSearch.investigationProbability
            if randomNumber <= EnvironmentSearch.investigationProbability:
                EnvironmentSearch.isInvestigation = True
                return

        EnvironmentSearch.isInvestigation = False

    @classmethod
    def dataAccumulation(cls, forward: int, investEpoch: bool, TD_target: tensor, Qprevious: tensor):
        """
        Аккумулирование данных в неисследовательскую эпоху.

        :param forward:
        :param investEpoch:
        :param TD_target:
        :param Qprevious:
        :return:
        """
        # if forward == 0:
        #     # Нулевой проход в каждой эпохе. Обнуляем аккумулированные значения
        #     EnvironmentSearch.TD_cumul = None
        #     EnvironmentSearch.Qpr_cumul = None

        if not investEpoch:
            # Во время эпохи без исследований, производим аккумулирование данных для эпохи исследований
            if EnvironmentSearch.TD_cumul is None:
                EnvironmentSearch.TD_cumul = TD_target
            else:
                EnvironmentSearch.TD_cumul += TD_target

            if EnvironmentSearch.Qpr_cumul is None:
                EnvironmentSearch.Qpr_cumul = Qprevious
            else:
                EnvironmentSearch.Qpr_cumul += Qprevious

    @classmethod
    def generate(cls, calc_device: device):
        """
        Возвращает сгенерированный выход актора для исследовательского прохода по критику

        :param calc_device:
        :return:
        """
        randomIndex = random.choice([0, 1, 2, 3, 4])
        randomBorder = random.uniform(0.5, 0.95)
        result = [random.uniform(0, randomBorder) for i in range(0, 5)]
        result[randomIndex] = random.uniform(randomBorder + 0.005, 1)

        target = [0, 0, 0, 0, 0]
        target[randomIndex] = 1

        return tensor([result], device=calc_device, dtype=float), tensor([target], device=calc_device, dtype=float)

    @classmethod
    def AccumulToNone(cls, forward: int, investEpoch: bool, batchCountinEpoch: int):
        """
        Собранные данные превращаются в None на нулевом проходе неисследовательской эпохи

        :param forward:
        :return:
        """
        # if (not investEpoch) and forward == 0:
        #     EnvironmentSearch.TD_cumul = None
        #     EnvironmentSearch.Qpr_cumul = None
        #     EnvironmentSearch.isInvestigation = False
        #     EnvironmentSearch.investigationProbability = 0.
        # else:
        #     EnvironmentSearch.investigationProbability = EnvironmentSearch.probability(batchCountinEpoch)

        if forward == 0:
            if investEpoch:
                EnvironmentSearch.investigationProbability = EnvironmentSearch.probability(batchCountinEpoch)
            else:
                EnvironmentSearch.TD_cumul = None
                EnvironmentSearch.Qpr_cumul = None
                EnvironmentSearch.isInvestigation = False
                EnvironmentSearch.investigationProbability = 0.
