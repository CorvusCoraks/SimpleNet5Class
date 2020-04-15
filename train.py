import net
import data
from torch.utils.data import DataLoader
from torch import device, cuda, nn, load, save, tensor, max, optim
import os
import tools
import td_zero


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

    learningRate = 0.0000001
    stopEpochNumber = 100000

    actorCreterion = nn.MSELoss()
    criticCreterion = nn.MSELoss()
    actorOptimizer = optim.Adam(netActor.parameters(), lr=learningRate)
    criticOptimizer = optim.Adam(netCritic.parameters(), lr=learningRate)


    if calc_device.type == 'cuda':
        netActor.to(calc_device)
        netCritic.to(calc_device)

    start_epoch = 0
    # Если существует файл с сохранённым состоянием нейронной сети, то загружаем параметры сети и тренировки из него
    if os.path.exists(actorCheckPointFile):
        checkpoint = load(actorCheckPointFile)
        # продолжаем обучение со следующей эпохи
        start_epoch = checkpoint['epoch'] + 1
        netActor.load_state_dict(checkpoint['state_dict'])
        actorOptimizer.load_state_dict(checkpoint['optimizer'])

    if os.path.exists(criticCheckPointFile):
        checkpoint = load(criticCheckPointFile)
        # продолжаем обучение со следующей эпохи
        start_epoch = checkpoint['epoch'] + 1
        netCritic.load_state_dict(checkpoint['state_dict'])
        criticOptimizer.load_state_dict(checkpoint['optimizer'])
        # net.eval()

    netActor.train()
    netCritic.train()

    reinf = td_zero.Reinforcement()
    td = td_zero.TD_zero(calc_device)

    # for m in netActor.modules():
    #     if isinstance(m, nn.Linear):
    #         print(m.weight)

    for name, module in netActor.named_modules():
        if name == '_ActorNet__h0Linear':
            print('Actor first hidden layer weights:\n', module.weight[0])
            break
    # print(netActor.modules())

    previousActorEpochLoss = 1000.0

    # количество правильных попаданий в класс за эпоху
    successByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
    # количество попаданий в каждый класс за эпоху (независимо от правильности)
    allByClasses = tensor([[0.001, 0.001, 0.001, 0.001, 0.001]], device=calc_device)
    # количество ошибочных попаданий в каждый класс (за эпоху)
    errorByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
    # outputByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)

    deltaV: float = 0.

    # investigationEpoch = False
    # investigationAction = False
    td_zero.EnvironmentSearch()

    # Необходимо создать этот тензор ВНЕ основного цикла, хотябы ввиде бутафории
    # criticInputs: tensor = tensor([[0]], requares_grad=True)
    for epoch in range(start_epoch, stopEpochNumber):
        td_zero.EnvironmentSearch.AccumulToNone(trainset.getTrainDataCount())
        actorEpochLoss = 0.0
        actorBatchLoss = 0.0
        i = 0
        print("Investigate Epoch: ", td_zero.EnvironmentSearch.isCuriosityEpoch())
        # почему-то при использовинии такого варианта, счётчик всегда равен нулю.
        for i, (actorInputs, actorTargets) in enumerate(trainloader, 0):
            actorOptimizer.zero_grad()
            criticOptimizer.zero_grad()
            # td_zero.EnvironmentSearch.AccumulToNone(i, investigationEpoch, trainset.getTrainDataCount())
            td_zero.EnvironmentSearch.setInvestigation(td_zero.EnvironmentSearch.isCuriosityEpoch())
            # criticInputs.zero
            if calc_device.type == 'cuda':
                actorInputs, actorTargets = actorInputs.to(calc_device), actorTargets.to(calc_device)
            actorOutputs = netActor(actorInputs)
            # print('Actor Inputs: ', actorInputs)
            # print('Actor Outputs: ', actorOutputs)
            # print('Actor Targets: ', actorTargets)

            if td_zero.EnvironmentSearch.isInvestigation:
                # если проход исследовательский
                (actorOutputs, _) = td_zero.EnvironmentSearch.generate(actorOutputs.device)

            rf = reinf.getReinforcement(actorOutputs, actorTargets)
            actorInputsList = actorInputs[0].tolist()
            actorInputsList.append(rf)

            criticInputs = tools.expandTensor(actorOutputs, actorInputsList)
            criticOutputs = netCritic(criticInputs)

            # Если это первый проход, то просто запоминаем предполагаемую величину ценнтости
            # Оптимизация проводится со второго прохода
            if td.getPreviousValue() is None:
                td.setPreviousValue(criticOutputs)
            else:
                previousValue = td.getPreviousValue()
                TDTarget = td.getTD(rf, criticOutputs)
                Vt = previousValue
                td_zero.EnvironmentSearch.dataAccumulation(i, td_zero.EnvironmentSearch.isCuriosityEpoch(), TDTarget,previousValue)
                # criticLoss = criticCreterion(previousValue, td.getTD(rf, criticOutputs))
                # ---
                # criticLoss = td_zero.AnyLoss.Qmaximization(previousValue, td.getTD(rf, criticOutputs))
                criticLoss = td_zero.AnyLoss.Qinverse(previousValue, td.getTD(rf, criticOutputs))
                td.setPreviousValue(criticOutputs)
                # ---
                criticLoss.backward()
                criticOptimizer.step()
                if not td_zero.EnvironmentSearch.isInvestigation:
                    actorOptimizer.step()
                    #investigationEpoch = True

                # for name, module in netActor.named_modules():
                #     if name == '_ActorNet__h0Linear':
                #         print(module.weight[0])

            (_, maxTargetIndex) = max(actorTargets, 1)
            (_, maxOutputIndex) = max(actorOutputs, 1)

            # outputByClasses[0, maxOutputIndex.item()] += 1
            if maxTargetIndex.item() == maxOutputIndex.item():
                successByClasses += actorTargets
            else:
                errorByClasses[0, maxOutputIndex.item()] += 1
            allByClasses += actorTargets

            # success = [successByClasses[0, i] / allByClasses[0, i] for i in range(successByClasses[0].size()[0])]
            # print('Success by classes: ', ['{0:6.3f}'.format(x) for x in success])
            # success = [successByClasses[0, i] / allByClasses[0, i] for i in range(successByClasses[0].size()[0])]
            # print('Success by classes: ', ['{0:6.3f}'.format(x) for x in outputByClasses[0]])
            # print(rf)

            actorLoss = actorCreterion(actorOutputs, actorTargets)

            # actorLoss.backward()
            # actorOptimizer.step()


            actorEpochLoss += actorLoss.item()
            actorBatchLoss += actorLoss.item()
            if (i+1) % 1000 == 0:  # print every 200 mini-batches
                print('[%d, %5d] mini-batch Actor avr loss: %.7f, TD target: %.7f, Value t: %.7f' %
                      (epoch, i+1, actorBatchLoss / 1000, TDTarget, Vt))
                # print('[{0:6.3f}, {1:6.3f}] mini-batch actor avr loss: {2:6.3f}; mini-batch critic avr loss: {3:6.3f}'.format(epoch, i, ))
                # actorBatchLoss = 0.0
                actorBatchLoss = 0.0

            if os.path.exists(actorCheckPointFile) and os.path.getsize(actorCheckPointFile) != 0:
                os.replace(actorCheckPointFile, actorCheckPointFile + '.bak')

            save({
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
        # Чередуем исследовательские и неисследовательские эпохи
        # investigationEpoch = True if investigationEpoch == False else False
        print('Ошибка эпохи: {}, Уменьшение ошибки эпохи: {}'.format(actorEpochLoss, previousActorEpochLoss - actorEpochLoss))
        success = [successByClasses[0, i] / allByClasses[0, i] for i in range(successByClasses[0].size()[0])]
        print('Epoch Success by classes: ', ['{0:6.4f}'.format(x) for x in success])
        fail = [errorByClasses[0, i] / allByClasses[0, i] for i in range(errorByClasses[0].size()[0])]
        print('Epoch Fail by classes: ', ['{0:6.4f}'.format(x) for x in fail])
        # error = [errorByClasses[0, i] / allByClasses[0, i] for i in range(errorByClasses[0].size()[0])]
        # print('Error by classes: ', ['{0:6.3f}'.format(x) for x in error])

        for name, module in netActor.named_modules():
            if name == '_ActorNet__h0Linear':
                print('Actor first hidden layer weights:\n', module.weight[0], '\n---')
                break

        successByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
        allByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)
        errorByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=calc_device)

        previousActorEpochLoss = actorEpochLoss

trainNet()