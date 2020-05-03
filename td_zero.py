from torch import tensor, max, device, float, inverse, sub, sqrt, mm, abs, exp, div
import random
from math import fabs


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
        self.__reinforcementByClass = [2., 2., -0.4, 2., 2.]
        # Наказание, если актор выбрал этот класс ошибочно
        self.__punishmentByClass = [-0.0, -0.0, -0.0, -0.0, -0.0]

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


class Reinforcement2:
    """
    Класс подкрепления, где наказание изначально уменьшено и увеличивается в процессе обучения
    """
    # Фунция НЕ ПРОХОДИЛА проверку!!!
    def __init__(self):
        # Подкрепление в случае удачного угадывания класса
        self.__reinforcementByClass = [2., 2., 2., 2., 2.]
        # Наказание, если актор выбрал этот класс ошибочно
        self.__punishmentByClass = [-0.2, -0.2, -0.2, -0.2, -0.2]
        self.__originPunishmentByClass = [-2, -2, -2, -2, -2]
        self.__startPunishmentByClass = [-0.2, -0.2, -0.2, -0.2, -0.2]
        # self.__punishmentStep = 0.01

    def punishmentCorrection(self, eraVt: float):
        pun = -1 - eraVt
        self.__punishmentByClass = [pun for i in range(len(self.__punishmentByClass))]

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


class Reinforcement3:
    """
    Подкрепление зависит от октивности исследования конкретного класса

    """
    # Подкрепление среды мы не трогаем.
    # Для каждого класса вводится поощеряющий мультипликатор за исследование этого класса, т. е. при исследовании
    # целевого класса, подкрепление среды усиливается подкреплением учителя ))), тем самым провоцируя актора использовать
    # неотработанные стратегии, т. е. проводить исследование.
    def __init__(self):
        # Подкрепление в случае удачного угадывания класса. В реальной системе, подкрепление является реакцией среды
        # и мы не знаем, какое подкрепление от среды на какое действие мы получим. Но в модели среды, подкрепления
        # и наказания заложены здесь
        self.__reinforcementByClass = [2., 2., 2., 2., 2.]
        # Наказание, если актор выбрал этот класс ошибочно
        self.__punishmentByClass = [-2., -2., -2., -2., -2.]
        # Мультипликатор подкрепления за проявленное любопытство.Особенно актуальнов начале обучения (исследование)
        self.__curiosityMulByClass = [1., 1., 1., 1., 1.]
        self.__curiosityMulZone = [-1, 1]
        self.__curiosityMulStep = 0.1
        # Мультипликатор смягчения наказания. Особенно актуально в начале обучения (во время исследования среды)
        self.__commutationMulByClass = [1., 1., 1., 1., 1.]
        self.__commutationMulZpne = [0, 1]
        self.__commutationMulStep = 0.1

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

        resultReinf = []
        # Заполняем результирующий массив подкреплений
        for i, value in enumerate(self.__reinforcementByClass):
            resultReinf.append(value * self.__curiosityMulByClass[i])

        if maxTargetIndex.item() == maxOutputIndex.item():
            # если действие актора правильное, т. е. совпадает с целью, то он по этому классу получает поощерение
            return tensor([[resultReinf[maxTargetIndex]]], dtype=float, device=outputs.device)
        else:
            # если актор не правильно поступил, то по этому же классу он получает наказание (по правильному классу)
            return tensor([[self.__punishmentByClass[maxTargetIndex]]], dtype=float, device=outputs.device)

    def setCuriosityMul(self, allByClassInEpoch: tensor):
        shotsNum = 0
        for each in allByClassInEpoch[0]:
            shotsNum += each

        avrByClass = shotsNum / len(allByClassInEpoch[0])
        # classesNum = len(allByClassInEpoch[0])

        for i, value in enumerate(allByClassInEpoch[0]):
            if value < avrByClass * 0.5:
                # если меньше половины среднего по классу
                # увеличиваем подкрепление
                if (self.__curiosityMulByClass[i] + self.__curiosityMulStep) <= self.__curiosityMulZone[1]:
                    self.__curiosityMulByClass[i] += self.__curiosityMulStep
            elif value > avrByClass * 1.5:
                # если больше полутора средних по классу
                # уменьшаем подкреление
                if (self.__curiosityMulByClass[i] - self.__curiosityMulStep) >= self.__curiosityMulZone[0]:
                    self.__curiosityMulByClass[i] -= self.__curiosityMulStep
            else:
                # подкрепление прежнее
                pass

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
    def Qinverse(cls, input: tensor, target: tensor, alpha=0.001):
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
        # Цифра 3 вычитается, чтобы сдвинуть график функции вправо, в плюсовую зону. Чтобы в пределах -1 ... 1
        # был более-менее видимый градиент
        return mm(sub(target, input), sub(target, input)) + alpha*inverse(exp(sub(input, 3)))


class EnvironmentSearch():
    """
    Класс исследования окружающей среды. Обеспечивает схему плавного перехода от чисто исселдовательских эпох к чисто
    обучающим и обратно. Отказался от использования в пользу исследований в рамках ВСЕХ эпох.
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
        if EnvironmentSearch.NumMap.epochType == EnvironmentSearch.NumMap.CURIOSITY:
            return True
        else:
            return False


class InEpochResearch():
    """
    Класс установки вида батча как исследовательского или простого обучающего.
    """
    def __init__(self):
        # вероятность того, что данный батч надо сделать исследоватлеьским
        # начальное значение -1 означает, что это - первая эра, первая эпоха обучения. Вобще, начало обучения.
        self.researchProbability: float = -1.
        # флаг, демонстрирующий, что данный батч исследовательский
        self.__isResearchBatch: bool = False

    def __recalculate(self):
        """
        Кидаем кости, меняя флаг исследовательскости

        :return:
        """
        if random.uniform(0, 1) <= self.researchProbability:
            self.__isResearchBatch = True
        else:
            self.__isResearchBatch = False

    def isResearchBatch(self, researchProbability=2, recalculate=False):
        """
        Данный батч исследовательский?

        :param researchProbability: Вероятность того, что данный батч исследовательский
        :param recalculate: Кидать кости?
        :return:
        :rtype bool:
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


class EpochData:
    """
    Класс-запись. Статистическая информация в рамках эпохи
    """
    def __init__(self, value=0., td=0.):
        """
        :param value: Vt
        :param td: TD-target
        """
        # собственно, запись данных
        self.avr = {'value': value, 'td-target': td}

    def addData(self, value: float, td: float):
        """
        Аккумулирование данных в записи.

        :param value: Vt
        :param td: TD-target
        """
        self.avr['value'] += value
        self.avr['td-target'] += td


class StatisticData:
    """
    Класс сбора статистической информации
    """
    def __init__(self, epochSize: int):
        """

        :param epochSize: количество данны х в эпохе (количество батчей * количество данных в батче)
        """
        self.__epochSize: int = epochSize
        # Глубина запоминания данных. Количество Эр
        self.__avrDataDeepInEra = 5
        # Глубина запоминания данных. Количество эпох
        self.__avrDataDeepInEpoch = 5
        # Аккумулятор. Буферная величина, в которой происходит суммирование/накопление данных в процессе эпохи
        self.__sumEpochData = EpochData()
        # Коллекция, где хранятся данные (усреднённые по размеру эпохи) по эпохам
        self.avrEpochCollection = [None for i in range(self.__avrDataDeepInEpoch)]
        # Коллекция. Хранит данные (усреднённые по размеру эпохи) ОДНОЙ ПОСЛЕДНЕЙ эпохи в рамках эры.
        self.avrEraCollection = [None for i in range(self.__avrDataDeepInEra)]

    def afterRestoreCheckPoint(self):
        # после восстановления из файла образа тренировки, нужно очистить этот аттрибут, чтобы при повторно мрохождении
        # эпохи, к данным прошлого прохода не суммировались данные нового прохода
        self.__sumEpochData = EpochData()

    def addBatchData(self, vt: tensor, td: tensor):
        """
        Добавление (приплюсовывание) данных в процессе прохождения эпохи

        :param vt: Vt
        :param td: TD-target
        """
        self.__sumEpochData.addData(vt[0].item(), td[0].item())

    def pushAvrEpochData(self):
        """
        Перенести накопленные за эпоху данные из аккумулятора в коллекцию.
        """
        # удаляем самое древнее значение
        self.avrEpochCollection.pop(0)
        # усредняем по эпохе
        self.__sumEpochData.avr['value'] = self.__sumEpochData.avr['value'] / self.__epochSize
        # усредняем по эпохе
        self.__sumEpochData.avr['td-target'] = self.__sumEpochData.avr['td-target'] / self.__epochSize
        # добавляем в конец
        self.avrEpochCollection.append(self.__sumEpochData)
        # очищаем буферный аккумулятор
        self.__sumEpochData = EpochData()

    def pushAvrEraData(self):
        """
        Перенести накопленные за эру данные из аккумулятора в коллекцию.
        """
        # удаляем самое древнее значение
        self.avrEraCollection.pop(0)
        # добавляем в конец самое свежее
        self.avrEraCollection.append(self.avrEpochCollection[len(self.avrEpochCollection) - 1])


class StatisticAnalitic:
    """
    Обработчик статистических данных. Буфер для их использования. Инструмент для работы с ними.
    """
    def __init__(self, data: StatisticData):
        self.__statisticData = data

    def avrEraChange(self, key: str, deep=3):
        """
        На сколько изменилось значение за послдение сохранённые эпохи

        :param deep: глубина (количество элементов в диапазоне)
        :return:
        :rtype float:
        """
        dataCollection = self.__statisticData.avrEraCollection
        if deep > len(dataCollection):
            # проверка диапазона
            return None
        elif dataCollection[-deep] is None or dataCollection[len(dataCollection)-1] is None:
            # проверка наличия данных
            return None
        else:
            return dataCollection[len(dataCollection) - 1].avr[key] - dataCollection[-deep].avr[key]

    def avrEpochChange(self, key: str, deep=3):
        """
        На сколько изменилось значение за послдение сохранённые эпохи

        :param key: ключ словаря, в котором хранится затребываемое значение
        :param deep: глубина (количество элементов в диапазоне, за который вычисляется относительное изменение величины)
        :return:
        :rtype float:
        """
        dataCollection = self.__statisticData.avrEpochCollection
        if deep > len(dataCollection):
            # проверка диапазона
            return None
        elif dataCollection[-deep] is None or dataCollection[len(dataCollection)-1] is None:
            # проверка наличия данных
            return None
        else:
            return dataCollection[len(dataCollection) - 1].avr[key] - dataCollection[-deep].avr[key]


class TrainingControl:
    """
    Управление процессом обучения
    """
    def __init__(self, analitic: StatisticAnalitic, deltaMin: float, deltaMax: float, deep: int):
        """

        :param analitic: Объект типа StatisticAnalitic
        :param deltaMin: Минимальное допустимое изменение Vt за deep Эр
        :param deltaMax: Максимально допустимое изменение Vt за deep Эр
        :param deep: Количество эр, по прошествии которых мы оцениваем изменение Vt
        """
        self.__analitic = analitic
        self.__deltaMin = deltaMin
        self.__deltaMax = deltaMax
        self.__deep = deep
        # self.__startAlpha = startAlpha

        # резервное значение времени моратория на изменение alpha
        # Когда принято решение на изменение alpha, считаем, что нам надо оценивать произведённый этим эфект уже
        # на новых данных, а значит, с новой alpha должны пройти deep Эр. Только после этого смотрим изменение Vt
        # и делаем выводы
        self.__waitBase = deep
        # счётчик Эр, отслеживающий количество пройденных Эр во время моратория
        self.waitCounter = 0

        # На переходе из Эры в Эру.
        # Маркер устанавливается в True, если мы запускаем проверочную эпоху (вероятность обучение = 0),
        # чтобы оценить достижения в процессе обучения завершившейся Эры.
        # После завершения проверочной эпохи, маркер устанавливается в False
        self.itWasSituationView = False
        # Теоретически, возможно, но маловероятно, прерывание обучения в середине тестовой эпохи и имеет смысл
        # сохнанить этот марке в файле вместе с остальными параметрами. Иначе, по возобновлении обучения, эпоха
        # проверки будет пропущена и сразу продолжится обучение.

    def throttle(self, startAlpha: float, currentAlpha: float, alphaStep=0.001):
        """
        Возвращает изменение значения коэффициента при обратной экспоненте

        :return: новое значения коэф. альфа
        :rtype float:
        """
        if 0 < self.waitCounter < self.__waitBase:
            # после изменения газа, несколько Эр подряд изменять его запрещено, чтобы получить обновлённую статистику
            self.waitCounter += 1
            print('Throttle function result: ', currentAlpha)
            return currentAlpha

        self.waitCounter = 0

        change = self.__analitic.avrEraChange('value', self.__deep)
        if change is None:
            # нет данных для управления, оставляем всё как было
            result = currentAlpha
        elif change > 0:
            if fabs(change) < self.__deltaMin:
                # добавляем газу
                self.waitCounter += 1
                result = currentAlpha + alphaStep
            elif fabs(change) > self.__deltaMax:
                # убавить газ
                if currentAlpha - alphaStep <= startAlpha:
                    # если при убирании газа мы уйдём ниже стартового альфа, то оставляем альфу такой какая она есть
                    # и счётчик ожидания не включаем
                    result = startAlpha
                else:
                    self.waitCounter += 1
                    result = currentAlpha - alphaStep
            else:
                result = currentAlpha
        else:
            # change <= 0
            # добавляем тягу
            self.waitCounter += 1
            result = currentAlpha + alphaStep

        print('Throttle function result: ', result)
        return result

    def probabilytyStep(self, research: InEpochResearch):
        """
        Вычисляет шаг изменения вероятности исследовательского батча в рамках Эпохи, в зависимости от стратегии обучения

        :param research:
        :return:
        :rtype float:
        """
        # С новой эпохой уменьшаем вероятность исследования на очередном батче
        if research.researchProbability <= 0.25:
            # 2 эпохи обучения
            return 0.1
        elif research.researchProbability <= 0.55:
            # 6 эпох обучения
            return 0.05
        #  от 0,95 до 0,55 - 40 эпох обучения
        return 0.01

    # def isSituationView(self):
    #
