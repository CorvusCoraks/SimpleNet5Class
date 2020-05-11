from torch import tensor, device, float, stack, squeeze
import random
import tools
import net
# from td_zero import Reinforcement

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

    # def getVariants(self, criticNet: net.CriticNet, actorInputs: tensor, actorTarget: tensor, reinforcement: Reinforcement, calcDevice: device):
    def getVariants(self, criticNet: net.CriticNet, actorInputs: tensor, actorTarget: tensor, reinforcement, calcDevice: device):
        """
        Для проведения исследовательских прогонов через критика

        :param criticNet: объект сети критика
        :param actorInputs: тензор [[...]], входов в актора
        :param actorTarget: тензор [[...]] правильных выходов для данного входа актора
        :param reinforcement: ФУНКЦИЯ рассчёта подкрепления
        :param calcDevice: устройство на котором идёт расчёт
        :return: [[], [], ...] - тензор подкреплений, [[], [], ...] - тензор функций ценности, [[], [], ...] - тензор
        вариантов выходов актора
        """
        #
        # предполагаем, что методы stack, squeeze, unsqueeze позволяют прямой и обратный проход
        #
        # количество выходных классов актора
        classCount = 5
        # лист исследовательских выходов актора
        actorOutputsList = []
        # лист тензоров выыодов критика
        criticInputsList = []
        for i in range(classCount):
            # Добавочный лист исследовательских пробных действий
            addList = [0 for i in range(classCount)]
            # ставим в позицию единицу, имитируя исследовательский выход актора
            addList[i] = 1
            actorOutputsList.append(addList)
            # заполняем лист тензоров расширенными входными тензорами актора
            criticInputsList.append(squeeze(tools.expandTensor(actorInputs, addList)))

        # Обратный проход здесь не нужен
        actorOutputs = tensor(actorOutputsList, dtype=float, requires_grad=False, device=calcDevice)

        # Превращаем лист входов критика в стопку из пяти тензоров
        criticInput = stack(criticInputsList)
        # Получаем тензор выходных оценок функции ценности
        criticOutput = criticNet(criticInput)

        # Лист подкреплений
        reinf = []
        for i in range(classCount):
            # заполняем лист подкреплений, согласно исследований актора
            actorOutputsOne = actorOutputs[i].clone().detach().unsqueeze(0)
            # reinf.append(squeeze(reinforcement.getReinforcement(actorOutputsOne, actorTarget), 0))
            reinf.append(squeeze(reinforcement(actorOutputsOne, actorTarget), 0))

        # и превращаем список в общий тензор
        reinf = stack(reinf)

        return reinf, criticOutput, actorOutputs


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