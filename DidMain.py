import torch
import torch.optim as optim

from DidModel import DidModel
from DidModelRunner import DidModelRunner
from ExampleDataset import ExampleDataset

if __name__ == "__main__":
    # get device on which training should run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define params for data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

    # load data
    csv_path = './data/dev/segmented/metadata.csv'
    file_path = './data/dev/segmented/'

    train_set = ExampleDataset(csv_path, file_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, **kwargs)

    # create our own model with classifier on top of fairseq's xlsr_53_56k.pt
    model = DidModel()

    # Define a Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # create runner for training and testing
    runner = DidModelRunner(device=device, model=model, optimizer=optimizer, scheduler=scheduler)

    log_interval = 20
    for epoch in range(1, 41):
        if epoch == 31:
            print("First round of training complete. Setting learn rate to 0.001.")
        scheduler.step()
        runner.train(train_loader=train_loader, epoch=epoch, log_interval=log_interval)
        # runner.test(test_loader=test_loader)
        # todo maybe safe model after every epoch

    print('Finished Training')
