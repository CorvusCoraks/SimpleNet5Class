from torch import tensor


def expandTensor(inputTensor: tensor, additional: list):
    """
    Добавление в конец тензора inputTensor данных из листа additional
    Предположительно, градиент обратного прохода будет считаться от конечного x к inputTensor
    :param inputTensor: тензор вида [[1, 2, 3]]
    :param additional: лист дополнительных величин вида [4, 5]
    :return: тензор вида [[1, 2, 3, 4, 5]]
    """
    blueprint = [1 for i in range(inputTensor.size()[1])]
    # Был лист [1, 1, 1] и длина листа addititional = 3,
    # в результате будет лист [1, 1, 4]
    blueprint[len(blueprint) - 1] += len(additional)
    new_tensor = tensor(blueprint, device=inputTensor.device)
    x = inputTensor.repeat_interleave(new_tensor, dim=1)
    j = 0
    for i in range(inputTensor.size()[1], x.size()[1]):
        x[0, i] = additional[j]
        j += 1
    return x

def expandTensor1d(inputTensor: tensor, additional: list):
    """
    Добавление в конец тензора inputTensor данных из листа additional

    Предположительно, градиент обратного прохода будет считаться от конечного x к inputTensor
    :param inputTensor: тензор вида [1, 2, 3]
    :param additional: лист дополнительных величин вида [4, 5]
    :return: тензор вида [1, 2, 3, 4, 5]
    """
    blueprint = [1 for i in range(inputTensor.size())]
    # Был лист [1, 1, 1] и длина листа addititional = 3,
    # в результате будет лист [1, 1, 4]
    blueprint[len(blueprint) - 1] += len(additional)
    new_tensor = tensor(blueprint, device=inputTensor.device)
    result = inputTensor.repeat_interleave(new_tensor, dim=1)
    j = 0
    for i in range(inputTensor.size(), result.size()):
        result[i] = additional[j]
        j += 1
    return result

# def random5generate():
#     pass