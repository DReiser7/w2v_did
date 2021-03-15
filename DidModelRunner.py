import torch.nn.functional as F


class DidModelRunner:

    def __init__(self, device, model, optimizer, scheduler):
        self.device = device
        print('running on device: ', self.device)

        self.model = model
        self.model = self.model.double()
        self.model.to(self.device)
        print('print model: ', self.model)

        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, train_loader, epoch, log_interval):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            target = target.to(self.device)
            data = data.requires_grad_()  # set requires_grad to True for training
            output = self.model(data)
            output = output['x']
            loss = F.nll_loss(output, target)  # the loss functions expects a batchSizex5 input
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:  # print training stats
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.detach().float()))

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            output = output['x']
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target).cpu().sum().item()
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                               100. * correct / len(test_loader.dataset)))