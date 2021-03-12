# !pip install fairseq
# !pip install torch
import torch
import fairseq
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F


from ExampleDataset import ExampleDataset
from DidModel import DidModel


def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    return [data, targets]


def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # data = data.to(device)
        # target = target.to(device)
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output['x'].permute(1, 0, 2) #original output dimensions are batchSizex1x10
        loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


if __name__ == "__main__":

    # cp_path = '../models/xlsr_53_56k.pt'  # TODO: https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
    # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    # model = model[0]
    # model.eval()

    model = DidModel()
    print(model)
    model = model.double()

    wav_input_16khz = torch.randn(1, 1000)
    # z = model.feature_extractor(wav_input_16khz)
    # c = model.feature_aggregator(z)

    # Freeze all the parameters in the network
    # for param in model.parameters():
    #     param.requires_grad = False

    csv_path = '../data/dev/wav/metadata.csv'
    file_path = '../data/dev/wav/'

    train_set = ExampleDataset(csv_path, file_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    log_interval = 20
    for epoch in range(1, 41):
        if epoch == 31:
            print("First round of training complete. Setting learn rate to 0.001.")
        scheduler.step()
        train(model, epoch)
        # test(model, epoch)

    print('Finished Training')
