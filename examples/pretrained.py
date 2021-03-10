# !pip install fairseq
# !pip install torch
import torch
import fairseq
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence


from ExampleDataset import ExampleDataset


def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    return [data, targets]


if __name__ == "__main__":

    cp_path = '../models/xlsr_53_56k.pt'  # TODO: https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    model = model[0]
    model.eval()

    print(model)

    wav_input_16khz = torch.randn(1, 1000)
    # z = model.feature_extractor(wav_input_16khz)
    # c = model.feature_aggregator(z)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    csv_path = '../data/dev/wav/metadata.csv'
    file_path = '../data/dev/wav/'

    train_set = ExampleDataset(csv_path, file_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.double()

    # Train the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.double())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
