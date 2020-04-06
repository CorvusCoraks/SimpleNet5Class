import net
import data
from torch.utils.data import DataLoader
from torch import device, cuda, nn, load, save, tensor, max, optim
import os

device = device("cuda:0" if cuda.is_available() else "cpu")
print(device)

batch_size = 1
trainset = data.TrainDataset()
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

check_point_file = '.\\actor.pth.tar'

netActor = net.ActorNet(True)

learningRate = 0.0001
stopEpochNumber = 100000

creterion = nn.MSELoss()
optimizer = optim.Adam(netActor.parameters(), lr=learningRate)

if device.type == 'cuda':
    netActor.to(device)

start_epoch = 0
# Если существует файл с сохранённым состоянием нейронной сети, то загружаем параметры сети и тренировки из него
if os.path.exists(check_point_file):
    checkpoint = load(check_point_file)
    # продолжаем обучение со следующей эпохи
    start_epoch = checkpoint['epoch'] + 1
    netActor.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

netActor.train()

previousActorEpochLoss = 1000.0

successByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
allByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=device)

for epoch in range(start_epoch, stopEpochNumber):
    epochLoss = 0.0
    batchLoss = 0.0
    i = 0
    # почему-то при использовинии такого варианта, счётчик всегда равен нулю.
    for i, (actorInputs, actorTargets) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        if device.type == 'cuda':
            actorInputs, actorTargets = actorInputs.to(device), actorTargets.to(device)
            actorOutputs = netActor(actorInputs)
            # print('Actor Inputs: ', actorInputs)
            # print('Actor Outputs: ', actorOutputs)
            # print('Actor Targets: ', actorTargets)
            (_, maxTargetIndex) = max(actorTargets, 1)
            (_, maxOutputIndex) = max(actorOutputs, 1)

            if maxTargetIndex.item() == maxOutputIndex.item():
                successByClasses += actorTargets
            allByClasses += actorTargets

            actorLoss = creterion(actorOutputs, actorTargets)
            actorLoss.backward()
            optimizer.step()
            epochLoss += actorLoss.item()
            batchLoss += actorLoss.item()
        if i % 1000 == 0 and i != 0:  # print every 200 mini-batches
            print('[%d, %5d] mini-batch Actor avr loss: %.7f' %
                  (epoch, i, batchLoss / 1000))
            # print('[{0:6.3f}, {1:6.3f}] mini-batch actor avr loss: {2:6.3f}; mini-batch critic avr loss: {3:6.3f}'.format(epoch, i, ))
            # actorBatchLoss = 0.0
            batchLoss = 0.0
    save({
        'epoch': epoch,
        'state_dict': netActor.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, check_point_file)

    print('Ошибка эпохи: {}, Уменьшение ошибки эпохи: {}'.format(epochLoss, previousActorEpochLoss - epochLoss))
    success = [successByClasses[0, i] / allByClasses[0, i] for i in range(successByClasses[0].size()[0])]
    print('Success by classes: ', ['{0:6.3f}'.format(x) for x in success])

    successByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    allByClasses = tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], device=device)

    previousActorEpochLoss = epochLoss