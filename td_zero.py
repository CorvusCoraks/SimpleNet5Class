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
        # если максимальное возможное подкрепление больше 1, то результат будет тоже больше единицы.
        # ВЫход сигмоиды не может быть
        # больше 1, а значит такой вариант ошибочен. Необходимо, чтобы сумма t+1 подкрепления и t+1 оценки функции
        # ценности ДОЛЖНЫ быть меньше единицы (так как фактически сравнивается с выходом критика).
        # Так же, при нулевом t+1 подкрелении, t+1 оценка функции ценности
        # сама по себе должна быть меньше единицы.
        #
        # Если установить, что выход критика это - 1/10 реальной функции ценности, то сумма t+1 подкрепления и
        # t+1 оценки функции ценности, для сравнения с выходом ктитика, тоже должна быть поделена на десять.
        # Следовательно. На практике, в формуле, нужно сравнивать сумму 1/10 подкрепления t+1 и t+1 выхода критика
        # с выходом ктитика t.
        result = reinforcement / 10 + self.__gamma * value_est - previousValue

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
        self.__reinforcementByClass = [2., 2., 2., 2., 2.]
        # Наказание, если актор выбрал этот класс ошибочно
        self.__punishmentByClass = [-2., -2., -2., -2., -2.]

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
            # если действие актора правильное, т. е. совпадает с целью, то он по этому классу получает поощерение
            return tensor([[self.__reinforcementByClass[maxTargetIndex]]], dtype=float, device=outputs.device)
        else:
            # если актор не правильно поступил, то по этому же классу он получает наказание (по правильному классу)
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
        :return: tensor [[scalar]]
        :rtype tensor:
        """
        # Добавлен обратная экспонента функции ценности. Цель - минимизация данного компонента должна вести
        # к максимизации функции ценности. Функция ценности умножена на число, призванное сделать обратную экспоненту
        # "покруче", т. е. не такой пологой, в диапазоне выхода сигмоиды
        return mm(sub(target, input), sub(target, input)) + 0.001*inverse(exp(sub(input, 3)))


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
    class NumMap():
        # последовательности эпох в цифровом виде. Элемент 0 - обучение, элемент 1 - исследование. Зациклено.
        # Фундаментальная карта. Эра, включающая в себя множество эпох. Когда в плане ниже, число исследовательских
        # эпох становится равным нулю, нижний план заменяется на этот, зацикливая процесс
        # Эра - последовательсность последовательность изменяющихся (внутренне перебаллансируемых) периодов
        # Период - набор эпох с внутренне нестабильным порядком типов эпох. Несколько периодов составляют Эру.
        #
        # 25 легче отслеживать в терминале
        baseMap = [1, 1]
        STUDY = 0
        CURIOSITY = 1
        # Тип текущей эпохи
        epochType = STUDY

        def __init__(self):
            # План эпох уровнем выше. После полного прохода нижнего плана, у этого плана учебные эпохи +1,
            # а исследовательские эпохи -1 (с целью переноса центра тяжести с исследования на обучение).
            # Далее, этот план копируется в нижний
            self.__epochMapNum = EnvironmentSearch.NumMap.baseMap.copy()
            # Рабочий план эпох. В процессе прохождения по циклу эпох, цифры в плане уменьшаются на 1. Когда элемент
            # становится равным нулю, происходит переход к следующему, и от последнего к первому
            self.__epochCounter = self.__epochMapNum.copy()

        def printMaps(self):
            print('Base Map: ', self.baseMap)
            print('Epoch Num Map: ', self.__epochMapNum)
            print('Epoch Counter: ', self.__epochCounter)

        def __setNextEra(self):
            """
            Переход к следующей эре.

            :return:
            """
            if self.__epochMapNum[EnvironmentSearch.NumMap.CURIOSITY] == 1:
                # Получаем сигнал, что очередная эра закончена и надо переходить к новой
                self.__epochMapNum = EnvironmentSearch.NumMap.baseMap.copy()

        def __changeEpochBalance(self):
            """
            Перебаллансировка типов эпох и начало нового периода.

            :return:
            """
            #
            # Перебаллансировка, это - увеличение числа обучающих эпох и уменьшение исследовательских эпох
            #
            if self.__epochCounter[EnvironmentSearch.NumMap.CURIOSITY] == 0:
                # Текущий период завершён
                if self.__epochMapNum[EnvironmentSearch.NumMap.CURIOSITY] == 1:
                    # Текущая эра завершена
                    self.__setNextEra()
                else:
                    # Производим перебаллансировку
                    self.__epochMapNum[EnvironmentSearch.NumMap.STUDY] += 1
                    self.__epochMapNum[EnvironmentSearch.NumMap.CURIOSITY] -= 1

        def setNextEpochType(self):
            # Обходим лист плана эпох
            if self.__epochCounter[EnvironmentSearch.NumMap.STUDY] == 0:
                # Если последовательность обучающих эпох пройдена
                if self.__epochCounter[EnvironmentSearch.NumMap.CURIOSITY] == 0:
                    self.__changeEpochBalance()
                    # Если последовательность исследовательских эпох пройдена
                    # Восстанавливаем изменяемый список
                    self.__epochCounter = self.__epochMapNum.copy()
                    # И объявляем эпоху обучающей
                    self.__epochCounter[EnvironmentSearch.NumMap.STUDY] -= 1
                    EnvironmentSearch.NumMap.epochType = EnvironmentSearch.NumMap.STUDY
                else:
                    # В противном случае, эпоха опять исследовательская
                    self.__epochCounter[EnvironmentSearch.NumMap.CURIOSITY] -= 1
                    EnvironmentSearch.NumMap.epochType = EnvironmentSearch.NumMap.CURIOSITY
            else:
                self.__epochCounter[EnvironmentSearch.NumMap.STUDY] -= 1
                EnvironmentSearch.NumMap.epochType = EnvironmentSearch.NumMap.STUDY

    def __init__(self):
        # TD-target накопленное за эпоху обучения, для использования в эпоху исследований
        self.__TD_cumul: tensor = None
        # накопленное за эпоху обучения значение Qt+1, для использования в эпоху исследований
        self.__Qpr_cumul: tensor = None
        # Флаг, показывающий, является ли текущий батч исследовательским
        self.isInvestigation: bool = False
        # вероятность того, что текущий батч будет исследовательским
        self.__investigationProbability = 0.

        self.__epochMap = EnvironmentSearch.NumMap()

    # Карта последовательных эпох: stady, curiosity, curiosity, ... Карта зациклена с последнего на первый эелемент
    # epochMap = ['s', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
    # Указатель внутри карты батчей
    # pointer = len(epochMap) - 1

    def __probability(self, batchCount: int):
        """
        Возвращает вероятность исследовательского прохода (батча)

        :param batchCount: количество батчей в эпохе
        :return: вероятность того, что данная эпоха исследовательская
        :rtype float:
        """
        # Функия явлеется целиком моей выдумкой. Основная цель, чтобы данная вероятность уменьшалась по мере
        # роста точности аппроксимации функции ценности.
        # Средняя разность целевой функции ценности и предсказанной на предыдущих батчах в рамках эпохи
        deltaMedium = div(sub(self.__TD_cumul, self.__Qpr_cumul),batchCount)
        # корень квадратный из квадрата средней разницы (чтоб убрать знак разницы и перейти к модулю)
        delta = sqrt(mm(deltaMedium, deltaMedium))
        # Нижеследующее является подгонкой. Чтобы в самом начале обучения, вероятность была не больше, но чуть
        # меньше единицы
        # if delta.item() * 10 >= 0.95:
        #     return 0.95
        # else:
        #     return delta.item() * 10

        return 0.95

    def setInvestigation(self, investEpoch: bool):
        """
        Будет ли производится ли на данном проходе (батче) исследование в рамках исследовательскойэпохи.

        :param investEpoch: данная эпоха исследовательская?
        """
        # Во входных данных вполне могут оказаться None
        if self.__TD_cumul is None or self.__Qpr_cumul is None:
            self.isInvestigation = False
            return

        if investEpoch:
            # Устанавливаем, будет ли данный батч в рамках исследовательской эпохи исследовательским
            randomNumber = random.uniform(0, 1)
            if randomNumber <= self.__investigationProbability:
                self.isInvestigation = True
                return

        self.isInvestigation = False

    def dataAccumulation(self, investEpoch: bool, TD_target: tensor, Qprevious: tensor):
        """
        Аккумулирование данных в неисследовательскую эпоху.

        :param investEpoch: эпоха инвестиционная?
        :param TD_target:
        :param Qprevious:
        """
        if not investEpoch:
            # Во время эпохи без исследований, производим аккумулирование данных для эпохи исследований
            if self.__TD_cumul is None:
                self.__TD_cumul = TD_target
            else:
                self.__TD_cumul += TD_target

            if self.__Qpr_cumul is None:
                self.__Qpr_cumul = Qprevious
            else:
                self.__Qpr_cumul += Qprevious

    def generate(self, calc_device: device):
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

    def accumulToNone(self, batchCountinEpoch: int):
        """
        Собранные данные превращаются в None на нулевом проходе обучающей эпохи

        :param batchCountinEpoch: размер одного батча
        :return:
        """
        self.__epochMap.setNextEpochType()

        # Устанавливаем начальные значения эпохи
        if self.isCuriosityEpoch():
            self.__investigationProbability = self.__probability(batchCountinEpoch)
            # print('************************* STUDY epoch. ******************************')
            print('Probability: ', self.__investigationProbability)
        else:
            self.__TD_cumul = None
            self.__Qpr_cumul = None
            self.isInvestigation = False
            self.__investigationProbability = 0.

    def isCuriosityEpoch(self):
        """
        Выясняет, является ли текущая эпоха исследовательской

        :return:
        """

        # if EnvironmentSearch.epochMap[EnvironmentSearch.pointer] == 's':
        #     return False
        # else:
        #     return True

        if EnvironmentSearch.NumMap.epochType == EnvironmentSearch.NumMap.CURIOSITY:
            return True
        else:
            return False


class InEpochResearch():
    def __init__(self):
        self.researchProbability: float = 0.
        self.__isResearchBatch: bool = False

    def __recalculate(self):
        # probability = random.uniform(0, 1)
        if random.uniform(0, 1) <= self.researchProbability:
            self.__isResearchBatch = True
        else:
            self.__isResearchBatch = False

    def isResearchBatch(self, researchProbability=2, recalculate=False):
        """
        Данный батч исследовательский?

        :param researchProbability:
        :param recalculate:
        :return:
        """
        if researchProbability <= 1:
            # если входная вероятность логична, то считаем её новой и пересчитываем состояние флага исследования
            self.researchProbability = researchProbability
            self.__recalculate()
        elif recalculate == True:
            # просто пересчитываем состояние флага на основе ранее сохранённой вероятности
            self.__recalculate()

        # возвращаем изменённое состояние после пересчёта, либо состояние старое, ранее сохранённое.
        return self.__isResearchBatch



    def generate(self, calc_device: device):
        """
        Возвращает сгенерированный выход актора для исследовательского прохода по критику

        :param calc_device: на чём происходит обсчёт тензоров
        :return: (сгенерированный случайный выход актора, соответствующий target актора) - тензоры [[...]]
        :rtype: tensor
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

