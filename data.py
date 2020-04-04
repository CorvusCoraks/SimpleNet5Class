import random
from torch.utils.data import Dataset
from torch import tensor
"""
Пара входов актора (всего 10 входов) соответствуют одному классу. Актор должен разбить поступившие сигналы на пять
классов (выходов)

На один из входов нейросети актора подаются сигнал 1, на все остальные ноль. Т. о. входной сигнал является членом
одного из пяти классов

"""

class NNData:
    def __init__(self):
        self.input = []
        self.output = []


class FiveClassDataset(Dataset):
    """
    Набор данных
    """
    def __init__(self, count_in_class: list):
        """

        :param count_in_class: количество блоков данных в каждом классе, лист вида [100, 10, 70]
        """
        # массив данных для нейросети
        self.__data = []
        # количество объектов в классах
        self.__countinclass = count_in_class

        random.seed(version=2)
        # заданное количество блоков данных в каждом классе
        # count_in_class = [10, 100, 1000, 100, 10]
        # в каждом классе находятся два вида блоков, согласно данному чертежу
        input_blueprint = [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]

        for i, value in enumerate(count_in_class):
            # обходим массив заданных количеств
            while value > 0:
                # пока в текущем классе ещё есть потребность в блоках
                # так как в каждом классе два вида входных данных, то случайным образом выбираем один из чертежей
                # данного класса
                index_in_class = random.choice([0, 1])
                # подготавливаем массив входных данных
                input = [0. for temp in range(len(input_blueprint[i][index_in_class]))]
                for j, blueprint in enumerate(input_blueprint[i][index_in_class]):
                    # согласно чертежу входного блока, заполняем массив входных данных
                    input[j] = random.uniform(0.01, 3) if blueprint == 1 else random.uniform(0, -3)

                # подготавливаем массив правильных выходов
                output = [0. for temp in range(len(count_in_class))]
                for k in range(len(count_in_class)):
                    # и заполняем его согласно текущего класса
                    output[k] = 1. if k == i else 0.

                data_block = NNData()
                data_block.input = input
                data_block.output = output

                # переносим полученный блок данных в общий массив
                self.__data.append(data_block)

                # уменьшаем потребное количество блоков в данном классе, так как создан новый блок
                value -= 1

        # перемешиваем полученный массив данных
        random.shuffle(self.__data)

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        return tensor(self.__data[item].input), tensor(self.__data[item].output)

    def probabilitys(self):
        """
        Возвращает лист вероятностей появления объекта по классам
        """
        sum = 0
        for value in self.__countinclass:
            sum += value

        result = []
        for i, value in enumerate(self.__countinclass):
            result.append(value/sum)

        return result



class TrainDataset(FiveClassDataset):
    def __init__(self):
        super(TrainDataset, self).__init__([100, 100, 100, 100, 100])


class TestDataset(FiveClassDataset):
    def __init__(self):
        super(TestDataset, self).__init__([100, 100, 100, 100, 100])