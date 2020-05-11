from torch import tensor, max, device, float, inverse, sub, mm, exp, detach
# import stats


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
        # result = reinforcement / 10 + self.__gamma * value_est - previousValue

        # подкрепление -2..2, выход выходного нейрона -1..1
        # в формуле делим подкрепление на 2, чтобы на выходе нейрона был максимум
        result = reinforcement / 2 + self.__gamma * value_est - previousValue
        # так как мы получаем гарантированную реакцию на каждом испытании, vslue_est на каждом ходе должно быть
        # равно нулю?
        # todo проверить
        # result = reinforcement / 2



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

    @classmethod
    def TD0_Grammon(cls, TDTarget: tensor, Qt: tensor):
        """
        Функция потерь на основании TD-Grammon
        :param TDTarget: rt+1 + gamma * Qt+1. НО! Возможно, максимальный вариант rt+1
        :param Qt: Qt
        :return: loss
        :rtype tensor:
        """
        # специальнео сделано, чтобы по разнице не пошло вычисление градиента
        deltaQ = sub(TDTarget, detach(Qt))
        # градиент будет вычисляться только по Qt
        return deltaQ.mm(Qt)


class TrainingControl:
    """
    Управление процессом обучения
    """
    # def __init__(self, analitic: StatisticAnalitic):
    def __init__(self):
        """

        :param analitic: Объект типа StatisticAnalitic
        :param deltaMin: Минимальное допустимое изменение Vt за deep Эр
        :param deltaMax: Максимально допустимое изменение Vt за deep Эр
        :param deep: Количество эр, по прошествии которых мы оцениваем изменение Vt
        """
        # self.__analitic = analitic
        # self.__deltaMin = deltaMin
        # self.__deltaMax = deltaMax
        # self.__deep = deep
        # self.__startAlpha = startAlpha

        # резервное значение времени моратория на изменение alpha
        # Когда принято решение на изменение alpha, считаем, что нам надо оценивать произведённый этим эфект уже
        # на новых данных, а значит, с новой alpha должны пройти deep Эр. Только после этого смотрим изменение Vt
        # и делаем выводы
        # self.__waitBase = deep
        # счётчик Эр, отслеживающий количество пройденных Эр во время моратория
        # self.waitCounter = 0

        # На переходе из Эры в Эру.
        # Маркер устанавливается в True, если мы запускаем проверочную эпоху (вероятность обучение = 0),
        # чтобы оценить достижения в процессе обучения завершившейся Эры.
        # После завершения проверочной эпохи, маркер устанавливается в False
        self.itWasSituationView = False
        # Теоретически, возможно, но маловероятно, прерывание обучения в середине тестовой эпохи и имеет смысл
        # сохнанить этот марке в файле вместе с остальными параметрами. Иначе, по возобновлении обучения, эпоха
        # проверки будет пропущена и сразу продолжится обучение.

    # def throttle(self, startAlpha: float, currentAlpha: float, alphaStep=0.001):
    #     """
    #     Возвращает изменение значения коэффициента при обратной экспоненте
    #
    #     :return: новое значения коэф. альфа
    #     :rtype float:
    #     """
    #     if 0 < self.waitCounter < self.__waitBase:
    #         # после изменения газа, несколько Эр подряд изменять его запрещено, чтобы получить обновлённую статистику
    #         self.waitCounter += 1
    #         print('Throttle function result: ', currentAlpha)
    #         return currentAlpha
    #
    #     self.waitCounter = 0
    #
    #     change = self.__analitic.avrEraChange('value', self.__deep)
    #     if change is None:
    #         # нет данных для управления, оставляем всё как было
    #         result = currentAlpha
    #     elif change > 0:
    #         if fabs(change) < self.__deltaMin:
    #             # добавляем газу
    #             self.waitCounter += 1
    #             result = currentAlpha + alphaStep
    #         elif fabs(change) > self.__deltaMax:
    #             # убавить газ
    #             if currentAlpha - alphaStep <= startAlpha:
    #                 # если при убирании газа мы уйдём ниже стартового альфа, то оставляем альфу такой какая она есть
    #                 # и счётчик ожидания не включаем
    #                 result = startAlpha
    #             else:
    #                 self.waitCounter += 1
    #                 result = currentAlpha - alphaStep
    #         else:
    #             result = currentAlpha
    #     else:
    #         # change <= 0
    #         # добавляем тягу
    #         self.waitCounter += 1
    #         result = currentAlpha + alphaStep
    #
    #     print('Throttle function result: ', result)
    #     return result

    def probabilytyStep(self, researchProbability: float):
        """
        Вычисляет шаг изменения вероятности исследовательского батча в рамках Эпохи, в зависимости от стратегии обучения

        :param researchProbability:
        :return:
        :rtype float:
        """
        # С новой эпохой уменьшаем вероятность исследования на очередном батче
        if researchProbability <= 0.25:
            # 2 эпохи обучения
            return 0.1
        elif researchProbability <= 0.55:
            # 6 эпох обучения
            return 0.05
        #  от 0,95 до 0,55 - 40 эпох обучения
        return 0.01

    # def isSituationView(self):
    #

