from torch import tensor, max, device, float, inverse, sub, sqrt, mm, abs, exp, div
import random


class TD_zero():
    def __init__(self, device: device, gamma=0.9):
        # коэф. приведения
        self.__gamma: float = gamma
        # Предыдущая предсказанная ценность
        self.__valuePrevious: tensor = None
        # Устройство, на котором идёт расчёт тензоров
        self.__device = device

    def getTD(self, reinforcement: tensor, value_est: tensor, TDTargetOnly=True):
        """
        Получить временную разницу

        :param reinforcement: подкрепление шага t+1
        :param value_est: прогнозная ценность на шаге t+1
        :param TDTargetOnly: использовать только TD-target (шага t+1), или TD-target - Vt
        :return:
        :rtype tensor:
        """
        previousValue = 0 if TDTargetOnly else self.__valuePrevious
        result = reinforcement + self.__gamma * value_est - previousValue

        # обновляем значение
        self.__valuePrevious = value_est

        # возвращаем результат, отсекая расчёт временных разниц от дерева вычислений (чтобы через него не было
        # обратного прохода
        return result.detach()

    def getPreviousValue(self):
        return self.__valuePrevious

    def setPreviousValue(self, previousValue: tensor):
        self.__valuePrevious = previousValue


class Reinforcement():
    """
    Класс подкрепления.
    """
    def __init__(self):
        # Подкрепление в случае удачного угадывания класса
        self.__reinforcementByClass = [5., 5., 5., 5., 5.]
        # Наказание, если актор выбрал этот класс ошибочно
        self.__punishmentByClass = [-0.005, -0.005, -0.005, -0.005, -0.005]

    def getReinforcement(self, outputs: tensor, targets: tensor):
        """
        Получить подкрепление.

        :param outputs: фактический выход актора
        :param targets: идеальный выход актора - цель
        :return:
        :rtype tensor:
        """
        (_, maxTargetIndex) = max(targets, 1)
        (_, maxOutputIndex) = max(outputs, 1)
        if maxTargetIndex.item() == maxOutputIndex.item():
            return tensor([[self.__reinforcementByClass[maxTargetIndex]]], dtype=float, device=outputs.device)
        else:
            return tensor([[self.__punishmentByClass[maxTargetIndex]]], dtype=float, device=outputs.device)


class AnyLoss():
    """
    Класс моих функций потерь
    """
    @classmethod
    def Qmaximization(cls, input: tensor, target: tensor):
        """
        MSELoss с добавленным обратный квадрат функции ценности.

        :param input: Предыдущее значение функции ценности
        :param target: TD-target
        :return:
        """
        # Добавлен обратный квадрат функции ценности. Цель - минимизация данного компонента должна вести
        # к максимизации функции ценности. НЕ РАБОТАЕТ
        return mm(sub(target, input), sub(target, input)) + inverse(mm(target, input))

    @classmethod
    def Qinverse(cls, input: tensor, target: tensor):
        """
        MSELoss c добавленной обратной экспонентой от функции ценности.

        :param input: Предыдущее значение функции ценности
        :param target: TD-target
        :return:
        """
        # Добавлен обратная экспонента функции ценности. Цель - минимизация данного компонента должна вести
        # к максимизации функции ценности. Функция ценности умножена на число, призванное сделать обратную экспоненту
        # "покруче", т. е. не такой пологой, в диапазоне выхода сигмоиды
        return mm(sub(target, input), sub(target, input)) + inverse(exp(input * 0.1))


class EnvironmentSearch():
    """
    Класс исследования окружающей среды.
    """
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
    # накопленное за эпоху значение TD-target
    #
    # Порядок вызова функций:
    # - начало прохода эпохам: AccumulToNone
    # - начало цикла прохода по эпохе: setInvestigation
    # - между проходом по актору и по критику в зависимости от isInvestigation производится generate или нет
    # - в зависимости от isInvestigation производится опримизация актора (в случае исследовательского прохода,
    # оптимизация не требуется).
    #
    # TD-target накопленное за эпоху обучения, для использования в эпоху исследований
    TD_cumul: tensor
    # накопленное за эпоху обучения значение Qt+1, для использования в эпоху исследований
    Qpr_cumul: tensor
    # Флаг, показывающий, является ли текущий батч исследовательским
    isInvestigation: bool = False
    # вероятность того, что текущий батч будет исследовательским
    investigationProbability = 0.

    # Карта последовательных батчей: stady, curiosity, curiosity, ... Карта зациклена с последнего на первый эелемент
    epochMap = ['s', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
    # Указатель внутри карты батчей
    pointer = len(epochMap) - 1

    @classmethod
    def probability(cls, batchCount: int):
        """
        Возвращает вероятность исследовательского прохода (батча)

        :param batchCount: количество батчей в эпохе
        :return: вероятность того, что данная эпоха исследовательская
        :rtype float:
        """
        # Функия явлеется целиком моей выдумкой. Основная цель, чтобы данная вероятность уменьшалась по мере
        # роста точности аппроксимации функции ценности.
        # Средняя разность целевой функции ценности и предсказанной на предыдущих батчах в рамках эпохи
        deltaMedium = div(sub(EnvironmentSearch.TD_cumul, EnvironmentSearch.Qpr_cumul),batchCount)
        # корень квадратный из квадрата средней разницы (чтоб убрать знак разницы и перейти к модулю)
        delta = sqrt(mm(deltaMedium, deltaMedium))
        # Нижеследующее является подгонкой. Чтобы в самом начале обучения, вероятность была не больше, но чуть
        # меньше единицы
        if delta.item() / 5 >= 0.95:
            return 0.95
        else:
            return delta.item() / 5

    @classmethod
    def setInvestigation(cls, investEpoch: bool):
        """
        Будет ли производится ли на данном проходе (батче) исследование в рамках исследовательскойэпохи.

        :param investEpoch: данная эпоха исследовательская?
        """
        # Во входных данных вполне могут оказаться None
        if EnvironmentSearch.TD_cumul is None or EnvironmentSearch.Qpr_cumul is None:
            EnvironmentSearch.isInvestigation = False
            return

        if investEpoch:
            # Устанавливаем, будет ли данный батч в рамках исследовательской эпохи исследовательским
            randomNumber = random.uniform(0, 1)
            if randomNumber <= EnvironmentSearch.investigationProbability:
                EnvironmentSearch.isInvestigation = True
                return

        EnvironmentSearch.isInvestigation = False

    @classmethod
    def dataAccumulation(cls, forward: int, investEpoch: bool, TD_target: tensor, Qprevious: tensor):
        """
        Аккумулирование данных в неисследовательскую эпоху.

        :param forward: номер батча в рамках эпохи
        :param investEpoch: эпоха инвестиционная?
        :param TD_target:
        :param Qprevious:
        """
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

        :param calc_device: на чём происходит обсчёт тензоров
        :return: (сгенерированный случайный выход актора, соответствующий target актора) - тензоры [[...]]
        """
        # индекс "правильного" класса
        randomIndex = random.choice([0, 1, 2, 3, 4])
        # граница, выше которой будет выход нейрона "правильного" класса, а ниже этой границы будут выходы
        # неправильных классов
        randomBorder = random.uniform(0.5, 0.95)
        # Генерируем выходы ниже гранцы по всем классам
        result = [random.uniform(0, randomBorder) for i in range(0, 5)]
        # для "правильного" класса генерируем выход выше границы
        result[randomIndex] = random.uniform(randomBorder + 0.005, 1)

        target = [0, 0, 0, 0, 0]
        target[randomIndex] = 1

        return tensor([result], device=calc_device, dtype=float), tensor([target], device=calc_device, dtype=float)

    @classmethod
    def AccumulToNone(cls, batchCountinEpoch: int):
        """
        Собранные данные превращаются в None на нулевом проходе неисследовательской эпохи

        :param forward:
        :return:
        """
        # Переводим указатель карты эпох к следующей
        if EnvironmentSearch.pointer == len(EnvironmentSearch.epochMap) - 1:
            EnvironmentSearch.pointer = 0
        else:
            EnvironmentSearch.pointer += 1

        # Устанавливаем начальные значения эпохи
        if EnvironmentSearch.isCuriosityEpoch():
            EnvironmentSearch.investigationProbability = EnvironmentSearch.probability(batchCountinEpoch)
            print('Probability: ', EnvironmentSearch.investigationProbability)
        else:
            EnvironmentSearch.TD_cumul = None
            EnvironmentSearch.Qpr_cumul = None
            EnvironmentSearch.isInvestigation = False
            EnvironmentSearch.investigationProbability = 0.

    @classmethod
    def isCuriosityEpoch(cls):
        """
        Устанавливает, является ли текущая эпоха исследовательской

        :return:
        """

        if EnvironmentSearch.epochMap[EnvironmentSearch.pointer] == 's':
            return False
        else:
            return True

