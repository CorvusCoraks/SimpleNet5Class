import net
import data
from torch.utils.data import DataLoader
from torch import device, cuda, nn, load, save, tensor, max, optim
import os
import tools
import td_zero
import winsound


def trainNet(savePath='.\\', actorCheckPointFile='actor.pth.tar', criticCheckPointFile='critic.pth.tar'):
    calc_device = device("cuda:0" if cuda.is_available() else "cpu")
    print(calc_device)

    batch_size = 1
    trainset = data.TrainDataset()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    actorCheckPointFile = savePath + actorCheckPointFile
    criticCheckPointFile = savePath + criticCheckPointFile

    print(actorCheckPointFile, ' || ', criticCheckPointFile)

    # def setPathsInColab(colabActorFile, colabCriticactor=actorCheckPointFile, critic=criticCheckPointFile):


    netActor = net.ActorNet(True)
    netCritic = net.CriticNet(True)

    learningRate = 0.000001
    stopEpochNumber = 100000
    epochSamples = 5000
    # начать с 0,001
    Vt_coef = 0.001
    vtCoefStart = 0.001
    # для справки. Изменение alpha даже на 0,001 может вызывать колебания (как минимум первоначальные) Vt до 0,1
    # Так что, с шагом и величиной alpha надо быть ОЧЕНЬ аккуратным
    VtCoefStep = 0.001

    actorCreterion = nn.MSELoss()
    criticCreterion = nn.MSELoss()
    actorOptimizer = optim.Adam(netActor.parameters(), lr=learningRate)
    criticOptimizer = optim.Adam(netCritic.parameters(), lr=learningRate)


    if calc_device.type == 'cuda':
        netActor.to(calc_device)
        netCritic.to(calc_device)

    beginProbability = 0.95
    # cardMap = {'0.95': 0.001, '0.65': 0.05, '0.30': 0.1}
    research = td_zero.InEpochResearch()
    # research.isResearchBatch(researchProbability=beginProbability)
    # research.researchProbability
    # research.isResearchBatch(researchProbability=beginProbability)
    # statisticData = td_zero.TrainingDataCollecting(epochSamples)
    statisticInfo = td_zero.StatisticData(epochSamples)
    # Счётчик ожидания разрешения на смену альфы
    waitCounter = 0
    # Флаг. Текущая эпоха является контролькной?
    itWasSituationView = False
    # флаг, текущие данные загружены из чекпоинта?
    isRestoredTraining = False
    # envSearch = td_zero.EnvironmentSearch()
    start_epoch = 0
    # Если существует файл с сохранённым состоянием нейронной сети, то загружаем параметры сети и тренировки из него
    if os.path.exists(actorCheckPointFile):
        checkpoint = load(actorCheckPointFile)
        # envSearch = checkpoint['environmentResearch']
        # if checkpoint.get('statistic') != None:
        Vt_coef = checkpoint['valueAlpha']
        # statisticData = checkpoint['statistic']
        statisticInfo = checkpoint['statistic']
        waitCounter = checkpoint['waitCounter']
        itWasSituationView = checkpoint['isSituationView']
        research = checkpoint['environmentResearch']
        # продолжаем обучение со следующей эпохи
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = checkpoint['epoch']
        netActor.load_state_dict(checkpoint['state_dict'])
        actorOptimizer.load_state_dict(checkpoint['optimizer'])
        if actorOptimizer.param_groups[0]['lr'] != learningRate:
            actorOptimizer.param_groups[0]['lr'] = learningRate

    if os.path.exists(criticCheckPointFile):
        checkpoint = load(criticCheckPointFile)
        # продолжаем обучение со следующей эпохи
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = checkpoint['epoch']
        netCritic.load_state_dict(checkpoint['state_dict'])
        criticOptimizer.load_state_dict(checkpoint['optimizer'])
        if criticOptimizer.param_groups[0]['lr'] != learningRate:
            criticOptimizer.param_groups[0]['lr'] = learningRate
        isRestoredTraining = True
        # net.eval()

    statisticInfo.afterRestoreCheckPoint()
    analitic = td_zero.StatisticAnalitic(statisticInfo)
    driver = td_zero.TrainingControl(analitic, 0.02, 0.03, 3)
    driver.waitCounter = waitCounter
    driver.itWasSituationView = itWasSituationView

    netActor.train()
    netCritic.train()

    reinf = td_zero.Reinforcement()
    td = td_zero.TD_zero(calc_device)

    for name, module in netActor.named_modules():
        if name == '_ActorNet__h0Linear':
            print('Actor first hidden layer weights:\n', module.weight[0])
        # elif name == '_ActorNet__outLinear':
        #     print('Actor out layer weights:\n', module.weight, '\n---')

    previousActorEpochLoss = 1000.0

    # количество правильных попаданий в класс за эпоху
    successByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
    # количество попаданий в каждый класс за эпоху (независимо от правильности)
    allByClasses = tensor([[0.001, 0.001, 0.001, 0.001, 0.001]], device=calc_device)
    # количество ошибочных попаданий в каждый класс (за эпоху)
    errorByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)

    # если у нас есть сохранённая тренировка, то номер стартовой эпохи увеличиваем на 1, чтобы продолжить со следующей
    # start_epoch = 0 if start_epoch == 0 else start_epoch + 1

    for epoch in range(start_epoch, stopEpochNumber):
        # envSearch.accumulToNone(trainset.getTrainDataCount())
        # envSearch._EnvironmentSearch__epochMap.printMaps()
        print('Critic Learning Rate: ', criticOptimizer.param_groups[0]['lr'])
        actorEpochLoss = 0.0
        actorBatchLoss = 0.0
        # old_alpha = 0.
        # запоминаем неизменённую текущую альфу, чтобы иметь возможность записать её в чекпоинт
        # и чтобы после возможного восстановления из чекпоинта вернутся в начало эпохи
        old_alpha = Vt_coef
        # запоминаем и старый счётчик ожидания. Если прервётся данная эпоха, восстановимся с этого значения из чекпоинта
        old_WaitingCounter = driver.waitCounter

        if not isRestoredTraining:
            # если это заход не с чекпоинта, смело меняем параметры обучения
            probabilityStep = driver.probabilytyStep(research)
            newProbability = research.researchProbability - probabilityStep

            if newProbability < 0.:
                # если новая вероятность ушла в минус, значит переход из Эры в Эру.
                if driver.itWasSituationView:
                    # Предыдущая эпоха была эпохохой оценки достижений, просмотром ситуации.
                    statisticInfo.pushAvrEraData()
                    # Возвращаемся к обычному обучению. Закольцовываем на начало новой Эры
                    driver.itWasSituationView = False
                    newProbability = beginProbability
                    # Изменяем коэф. альфа
                    Vt_coef = driver.throttle(vtCoefStart, Vt_coef, VtCoefStep)

                    # звуковой сигнал
                    frequency = 70  # Set Frequency To 2500 Hertz
                    duration = 5000  # Set Duration To 1000 ms == 1 second
                    winsound.Beep(frequency, duration)

                    input('Продолжить...')
                else:
                    # Переход из Эры в Эру. На переходе делаем эпоху оценки достижений
                    # Средние данные по эпохе на этом проходе попадут в коллекцию эпох, а на следующем проходе
                    # буду пушены в коллекцию эр (перед изменением коэф. альфа)
                    print('Check Era study situation.')
                    driver.itWasSituationView = True
                    newProbability = 0.
            research.isResearchBatch(researchProbability=newProbability)
        # восстанавливаем значение флага
        isRestoredTraining = False

        print('Research Probability: ', research.researchProbability)
        i = 0
        # почему-то при использовинии такого варианта, счётчик всегда равен нулю.
        for i, (actorInputs, actorTargets) in enumerate(trainloader, 0):
            actorOptimizer.zero_grad()
            criticOptimizer.zero_grad()

            if calc_device.type == 'cuda':
                actorInputs, actorTargets = actorInputs.to(calc_device), actorTargets.to(calc_device)
            actorOutputs = netActor(actorInputs)
            # print('Actor Inputs: ', actorInputs)
            # print('Actor Outputs: ', actorOutputs)
            # print('Actor Targets: ', actorTargets)

            # if envSearch.isInvestigation:
            if research.isResearchBatch(recalculate=True):
                # если проход исследовательский
                # (actorOutputs, _) = envSearch.generate(actorOutputs.device)
                (actorOutputs, _) = research.generate(actorOutputs.device)

            rf = reinf.getReinforcement(actorOutputs, actorTargets)
            actorInputsList = actorInputs[0].tolist()
            # actorInputsList.append(rf)

            criticInputs = tools.expandTensor(actorOutputs, actorInputsList)
            criticOutputs = netCritic(criticInputs)

            # Если это первый проход, то просто запоминаем предполагаемую величину ценнтости
            # Оптимизация проводится со второго прохода
            if td.getPreviousValue() is None:
                td.setPreviousValue(criticOutputs)
            else:
                TDTarget = td.getTD(rf, criticOutputs)
                Vt = td.getPreviousValue()

                statisticInfo.addBatchData(Vt, TDTarget)
                criticLoss = td_zero.AnyLoss.Qinverse(Vt, TDTarget, Vt_coef)
                td.setPreviousValue(criticOutputs)
                criticLoss.backward()
                criticOptimizer.step()
                if not research.isResearchBatch():
                    actorOptimizer.step()

            (_, maxTargetIndex) = max(actorTargets, 1)
            (_, maxOutputIndex) = max(actorOutputs, 1)

            # outputByClasses[0, maxOutputIndex.item()] += 1
            if maxTargetIndex.item() == maxOutputIndex.item():
                successByClasses += actorTargets
            else:
                errorByClasses[0, maxOutputIndex.item()] += 1
            allByClasses += actorTargets

            actorLoss = actorCreterion(actorOutputs, actorTargets)

            actorEpochLoss += actorLoss.item()
            actorBatchLoss += actorLoss.item()
            if (i+1) % 1000 == 0:  # print every 200 mini-batches
                print('[%d, %5d] mini-batch Actor avr loss: %.7f, TD target: %.7f, Value t: %.7f, criticLoss: %.7f' %
                      (epoch, i+1, actorBatchLoss / 1000, TDTarget, Vt, criticLoss))
                actorBatchLoss = 0.0

                if os.path.exists(actorCheckPointFile) and os.path.getsize(actorCheckPointFile) != 0:
                    os.replace(actorCheckPointFile, actorCheckPointFile + '.bak')

                save({
                    # 'environmentResearch': envSearch,
                    'valueAlpha': old_alpha,
                    'statistic': statisticInfo,
                    'waitCounter': old_WaitingCounter,
                    'isSituationView': driver.itWasSituationView,
                    # 'analitic': analitic,
                    # 'driver': driver,
                    'environmentResearch': research,
                    'epoch': epoch,
                    'state_dict': netActor.state_dict(),
                    'optimizer': actorOptimizer.state_dict(),
                }, actorCheckPointFile)

                if os.path.exists(criticCheckPointFile) and os.path.getsize(criticCheckPointFile) != 0:
                    os.replace(criticCheckPointFile, criticCheckPointFile + '.bak')

                save({
                    'epoch': epoch,
                    'state_dict': netCritic.state_dict(),
                    'optimizer': criticOptimizer.state_dict(),
                }, criticCheckPointFile)
        # запечатываем наколенные за эпоху данные
        # statisticInfo.closeEpochData()
        statisticInfo.pushAvrEpochData()
        # Чередуем исследовательские и неисследовательские эпохи
        # investigationEpoch = True if investigationEpoch == False else False
        forPrint1 = statisticInfo.avrEpochCollection[len(statisticInfo.avrEpochCollection)-1]
        forPrint2 = analitic.avrEpochChange('value') if analitic.avrEpochChange('value') is not None else 0.
        print('Epoch TD target: {0:6.4f}, Vt: {1:6.4f}, Vt change for 3 epoch: {2:6.4f}, alpha: {3:6.4f}'.
              format(forPrint1.avr['td-target'],
                        forPrint1.avr['value'],
                        forPrint2,
                        Vt_coef))
        # print('Epoch TD target: {0:6.4f}, Vt: {1:6.4f}, Vt change for 3 epoch: {2:6.4f}'.
        #       format(statisticData.getAvrTDtarget()[0].item(),
        #             statisticData.getAvrVt()[0].item(),
        #             statisticData.get3epochVT()[0].item()))
        print('Ошибка эпохи: {}, Уменьшение ошибки эпохи: {}'.format(actorEpochLoss, previousActorEpochLoss - actorEpochLoss))
        success = [successByClasses[0, i] / allByClasses[0, i] for i in range(successByClasses[0].size()[0])]
        print('Epoch real success action by classes: ', ['{0:6.4f}'.format(x) for x in success])
        fail = [errorByClasses[0, i] / allByClasses[0, i] for i in range(errorByClasses[0].size()[0])]
        print('Epoch real fail action by classes: ', ['{0:6.4f}'.format(x) for x in fail])
        # error = [errorByClasses[0, i] / allByClasses[0, i] for i in range(errorByClasses[0].size()[0])]
        # print('Error by classes: ', ['{0:6.3f}'.format(x) for x in error])

        for name, module in netActor.named_modules():
            if name == '_ActorNet__h0Linear':
                print('Actor first hidden layer weights:\n', module.weight[0], '\n---')
            # elif name == '_ActorNet__outLinear':
            #     print('Actor out layer weights:\n', module.weight, '\n---')

        successByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
        allByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
        errorByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)

        previousActorEpochLoss = actorEpochLoss

        # if epoch == 0:
        #     # Прошедшая эпоха была нулевой, где мы смотрели чистый выход сетей без исследований.
        #     # Но в дальнейшем, начинаются эпохи с исследованиями. УРА!
        #     research.researchProbability = beginProbability

trainNet()