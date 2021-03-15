import torch
import torch.optim as optim

from DidDataset import DidDataset
from DidModel import DidModel
from DidModelRunner import DidModelRunner

if __name__ == "__main__":
    # get device on which training should run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define params for data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

    # build train data
    csv_path_train = './data/dev/segmented/metadata.csv'
    file_path_train = './data/dev/segmented/'

    train_set = DidDataset(csv_path_train, file_path_train)
    print("Train set size: " + str(len(train_set)))

    # build test data
    csv_path_test = './data/dev/segmented/metadata.csv'
    file_path_test = './data/dev/segmented/'

    test_set = DidDataset(csv_path_test, file_path_test)
    print("Test set size: " + str(len(test_set)))

    # build data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, **kwargs)

    # create our own model with classifier on top of fairseq's xlsr_53_56k.pt
    model = DidModel(num_classes=5, freeze_fairseq=True)

    # Define a Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # create runner for training and testing
    runner = DidModelRunner(device=device, model=model, optimizer=optimizer, scheduler=scheduler)

    log_interval = 20
    for epoch in range(1, 41):
        if epoch == 31:
            print("First round of training complete. Setting learn rate to 0.001.")
        runner.train(train_loader=train_loader, epoch=epoch, log_interval=log_interval)
        runner.test(test_loader=test_loader)
        scheduler.step()
        if epoch % 5 == 0:  # save model every 5 epochs
            model_path = './models/did_model_epoch_' + str(epoch) + '.pt'
            print("Saving model to " + model_path)
            torch.save(model.state_dict(), model_path)

    print('Finished Training')
