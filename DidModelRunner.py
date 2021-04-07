import time
import torch
from transformers import Wav2Vec2FeatureExtractor

class DidModelRunner:

    def __init__(self, device, model, optimizer, scheduler, wandb, loss_function):
        self.device = device
        self.wandb = wandb
        print('running on device: ', self.device)

        self.model = model
        self.model = self.model.double()
        self.model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function

        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                          do_normalize=True, return_attention_mask=True)

    def train(self, train_loader, epoch, log_interval):
        self.model.train()
        closs = 0
        t = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            self.wandb.log({"dataload_duration": (time.time() - t)})
            data = torch.from_numpy(self.feature_extractor(data, sampling_rate=16000).input_values)
            data = data.to(self.device)
            target = target.to(self.device)
            z = time.time()
            output = self.model(data)['normalized']
            self.wandb.log({"model_calc_duration": (time.time() - z)})
            z = time.time()
            loss = self.loss_function(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.wandb.log({"batch_loss_calc_duration": (time.time() - z)})
            self.wandb.log({"batch_loss": loss.detach().item()})
            closs = closs + loss.detach().item()
            if batch_idx % log_interval == 0:  # print training stats
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.detach().float()))

            t = time.time()
        return closs


    def test(self, test_loader, log_interval):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            vloss = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                z = time.time()
                output = self.model(data)
                self.wandb.log({"eval_model_calc_duration": (time.time() - z)})
                # z = time.time()
                # loss = self.loss_function(output['normalized'], target)  # the loss functions expects a batchSizex5 input
                # vloss = vloss + loss.detach().item()
                # self.wandb.log({"eval_loss_calc_duration": (time.time() - z)})
                pred = output['x'].max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target).cpu().sum().item()
                if batch_idx % log_interval == 0:  # print training stats
                    print('Eval done: {:.0f}%'.format(100. * batch_idx / len(test_loader)))

            accr = 100. * correct / len(test_loader.dataset)
            self.wandb.log({"accuracy": accr})
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), accr))
            return vloss
