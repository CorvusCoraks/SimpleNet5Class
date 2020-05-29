import net
import data
from torch.utils.data import DataLoader
from torch import device, cuda, nn, load, save, tensor, max, optim, squeeze, unsqueeze, autograd
import os
import tools
import td_zero
import winsound
import stats
import random


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

    autograd.set_detect_anomaly(True)

    netActor = net.ActorNet(True)
    netCritic = net.CriticNet(True)

    learningRate = 0.000001
    stopEpochNumber = 100000
    epochSamples = 5000

    actorCreterion = nn.MSELoss()
    criticCreterion = nn.MSELoss()
    actorOptimizer = optim.Adam(netActor.parameters(), lr=learningRate)
    criticOptimizer = optim.Adam(netCritic.parameters(), lr=learningRate)


    if calc_device.type == 'cuda':
        netActor.to(calc_device)
        netCritic.to(calc_device)

    # beginProbability = 0.95
    # cardMap = {'0.95': 0.001, '0.65': 0.05, '0.30': 0.1}
    research = stats.InEpochResearch()
    # research.isResearchBatch(researchProbability=beginProbability)
    # research.researchProbability
    # research.isResearchBatch(researchProbability=beginProbability)
    # statisticData = td_zero.TrainingDataCollecting(epochSamples)
    statisticInfo = stats.StatisticData(epochSamples)
    # Счётчик ожидания разрешения на смену альфы
    # waitCounter = 0
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
        # Vt_coef = checkpoint['valueAlpha']
        # statisticData = checkpoint['statistic']
        statisticInfo = checkpoint['statistic']
        # waitCounter = checkpoint['waitCounter']
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
    analitic = stats.StatisticAnalitic(statisticInfo)
    driver = td_zero.TrainingControl()
    # driver.waitCounter = waitCounter
    driver.itWasSituationView = itWasSituationView

    netActor.train()
    netCritic.train()

    reinf = td_zero.Reinforcement()
    td = td_zero.TD_zero(calc_device)
    reinf5 = td_zero.Reinforcement5()

    # test = reinf.getReinforcement
    # print(type(test))

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

            # Получить подкрепление и значение оценки функции ценности для пяти вариантов действий
            # (reinf_tp1, Q_tp1, actorCuriosityTargets) = \
            #     research.getVariants(netCritic, actorInputs, actorTargets, reinf, calc_device)
            (reinf_tp1, Q_tp1, actorCuriosityTargets) = \
                research.getVariants(netCritic, actorInputs, actorTargets, reinf.getReinforcement, calc_device)
            # Предполагается, что максимальное подкрепление будет только в одном варианте
            # Величина максимального подкрепления и его индекс в общем тензоре тензоре
            # То есть, это результат ИДЕАЛЬНЫХ действий актора
            (maxReinf, maxRindex) = max(reinf_tp1, 0)
            # Одномерный тензор превращаем в двумерный
            maxReinf = unsqueeze(maxReinf, 0)
            maxRindex = maxRindex.item()
            # maxReinf = max(reinf_tp1)
            # Максимальная величина оценки и её индекс в общем тензоре
            (maxQtp1, maxQtp1Index) = max(Q_tp1, 0)
            maxQtp1 = unsqueeze(maxQtp1, 0)
            maxQtp1Index = maxQtp1Index.item()

            # Результат неидеальных действий актора, т. е. тех действий, которые будут продиктованы максимальной
            # величинеой функции оценки
            reinfByMaxQtp1 = unsqueeze(reinf_tp1[maxQtp1Index], 0)

            #
            # Стратегия. На каждом шаге выбирается вариант действий с максимальным Qtp1
            #

            # Если это первый проход, то просто запоминаем предполагаемую величину ценности
            # Оптимизация проводится со второго прохода
            if td.getPreviousValue() is None:
                # previous =
                td.setPreviousValue(maxQtp1)
            else:
                (step, rf5) = reinf5.getReinforcement()
                if step == 4:
                    pass
                else:
                    pass
                # Максимальная оценка Qt+1 должна сходиться к максимальному подкреплению rt+1 на каждом варианте
                # данных (так как различные варианты не связаны между собой причинно-следственной связью, в данном
                # случае)
                TDTarget = td.getTD(reinfByMaxQtp1, maxQtp1, useEstValue=False)
                Vt = td.getPreviousValue()

                statisticInfo.addBatchData(Vt, TDTarget)

                testRandom = random.uniform(0, 1)
                # testRandom = 0.4
                if testRandom < 0.4:
                    curiosityIndex = random.choice([0, 1, 2, 3, 4])
                    TDTarget = reinf_tp1[curiosityIndex]
                    TDTarget = unsqueeze(TDTarget, 0) / 2
                    Vt = Q_tp1[curiosityIndex]
                    Vt = unsqueeze(Vt, 0)
                    # Vt.unsqueeze_(0)
                    # TDTarget = reinfByMaxQtp1 / 2
                    # Vt = maxQtp1
                    actorCuriosityOutputs = actorCuriosityTargets[curiosityIndex].clone().detach().unsqueeze(0)
                else:
                    TDTarget = reinfByMaxQtp1 / 2
                    Vt = maxQtp1
                    actorCuriosityOutputs = actorCuriosityTargets[maxQtp1Index].clone().detach().unsqueeze(0)

                # TDTarget = reinfByMaxQtp1 / 2
                # Vt = maxQtp1
                # actorCuriosityOutputs = actorCuriosityTargets[maxQtp1Index].clone().detach().unsqueeze(0)

                actorLoss = actorCreterion(actorCuriosityOutputs, actorOutputs)


                criticLoss = criticCreterion(TDTarget, Vt)

                # criticLoss = td_zero.AnyLoss.TD0_Grammon(TDTarget, Vt)
                # criticLoss = criticCreterion(Vt, TDTarget)

                td.setPreviousValue(maxQtp1)

                criticLoss.backward()
                actorLoss.backward()

                criticOptimizer.step()
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
                    # 'valueAlpha': old_alpha,
                    'statistic': statisticInfo,
                    # 'waitCounter': old_WaitingCounter,
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
        # forPrint2 = analitic.avrEpochChange('value') if analitic.avrEpochChange('value') is not None else 0.
        print('Epoch TD target: {0:6.4f}, Vt: {1:6.4f}'.
              format(forPrint1.avr['td-target'],
                        forPrint1.avr['value']))
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

trainNet()