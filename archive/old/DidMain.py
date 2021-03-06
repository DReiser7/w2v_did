import json
import os
import sys
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import SubsetRandomSampler

import wandb
from DidDataset import DidDataset
from DidModel import DidModel
from DidModelClassifierOnly import DidModelClassifierOnly
from DidModelHuggingFaceOld import DidModelHuggingFaceOld
from DidModelRunner import DidModelRunner


def print_Config():
    print('data:')
    print("  train_data: " + config.data['train_dataset'])
    print("  test_data: " + config.data['test_dataset'])
    print("  batch_size: " + str(config.data['batch_size']) + ", shuffle: " + str(config.data['shuffle']))
    print('model:')
    try:
        print("  model: " + str(config.model['model_location']))
    except:
        print("  model: HuggingFace")
    print("  num_classes: " + str(config.model['num_classes']) + ", freeze_fairseq: " + str(
        config.model['freeze_fairseq']))
    print('general:')
    print("  num_workers: " + str(config.general['num_workers']))
    print("  epochs: " + str(config.general['epochs']))
    print("  optimizer: " + config.general['optimizer'])
    print("  loss_function: " + config.general['loss_function'])
    print("  log_interval: " + str(config.general['log_interval']) + ", model_save_interval: " + str(
        config.general['model_save_interval']))


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        did_config = json.load(f)

    # get device on which training should run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # batch_size has to stay the same even if we are on multiple gpus
    batch_size_test = did_config['data']['batch_size']
    # Using more than one GPU
    if torch.cuda.device_count() > 1:
        device_count = torch.cuda.device_count()
        print("Using:", device_count, "GPUs!")
        did_config['data']['batch_size'] = did_config['data']['batch_size'] * device_count
        batch_size_test = batch_size_test * device_count
        print("Multiplying batch * GPUs new batch_size=", did_config['data']['batch_size'])

    os.environ['WANDB_PROJECT'] = 'w2v_did'
    os.environ['WANDB_LOG_MODEL'] = 'true'

    # Initialize a new wandb run
    wandb.init(project='w2v_did', config=did_config, entity='ba-reisedomfiviapas',
               name=datetime.now().strftime("w2v_did " + "_%Y%m%d-%H%M%S"))
    # Config is a variable that holds and saves hyperparameters and inputs

    config = wandb.config
    print_Config()

    # files that match that pattern will save immediately once they're written to wandb.run.dir
    wandb.save("*.pt")

    # define params for data loaders
    kwargs = {'num_workers': config.general['num_workers'],
              'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

    # build train data
    csv_path_train = config.data['train_dataset'] + 'metadata.csv'  # file_path_train = './data/dev/segmented/'
    train_set = DidDataset(csv_path_train, config.data['train_dataset'])

    print("Train set size: " + str(len(train_set)))

    # build test data
    csv_path_test = config.data['test_dataset'] + 'metadata.csv'  # file_path_test = './data/dev/segmented/'
    test_set = DidDataset(csv_path_test, config.data['test_dataset'])

    test_set_indices = list(range(len(test_set)))
    np.random.shuffle(test_set_indices)
    val_split_index = int(np.floor(config.data['train_set_percentage'] * len(train_set)))
    test_idx = test_set_indices[:val_split_index]
    print("Test set size: " + str(len(test_idx)))

    # build data loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config.data['batch_size'],
                                               shuffle=config.data['shuffle'],
                                               drop_last=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size_test,
                                              sampler=SubsetRandomSampler(test_idx),
                                              drop_last=True,
                                              **kwargs)

    # Loss Function and fitting exponential normalizing function
    if config.general['loss_function'] == 'nllLoss':
        loss_function = F.nll_loss
        exp_norm_func = F.log_softmax
    else:
        raise SystemExit("you must specify loss_function for " + config.general['loss_function'])

    # create our own model with classifier on top of fairseq's xlsr_53_56k.pt
    try:
        model = DidModel(model_path=config.model['model_location'],
                         num_classes=config.model['num_classes'],
                         exp_norm_func=exp_norm_func,
                         freeze_fairseq=config.model['freeze_fairseq'])
    except:
        model = DidModelHuggingFaceOld(
            num_classes=config.model['num_classes'],
            exp_norm_func=exp_norm_func,
            freeze_fairseq=config.model['freeze_fairseq'])

    # Using more than one GPU
    if torch.cuda.device_count() > 1:
        print("Wrapping model with DataParallel")
        model = DataParallel(model)

    # Optimizer
    print('optimizer_params:')
    if config.general['optimizer'] == 'adam':
        print('  lr: ' + str(config.optimizers[config.general['optimizer']]['lr']) + ', weight_decay: ' + str(
            config.optimizers[config.general['optimizer']]['weight_decay']))

        if torch.cuda.device_count() > 1:
            optimizer = optim.Adam(model.parameters(),
                               lr=config.optimizers[config.general['optimizer']]['lr'],
                               weight_decay=config.optimizers[config.general['optimizer']]['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(),
                               lr=config.optimizers[config.general['optimizer']]['lr'],
                               weight_decay=config.optimizers[config.general['optimizer']]['weight_decay'])
    else:
        raise SystemExit("you must specify optimizer for " + config.general['optimizer'])

    # Scheduler
    print('scheduler_params:')
    print('  step_size: ' + str(config.scheduler['step_size']) + ', gamma: ' + str(config.scheduler['gamma']))
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.scheduler['step_size'],
                                          gamma=config.scheduler['gamma'])

    # create runner for training and testing
    runner = DidModelRunner(device=device,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            wandb=wandb,
                            loss_function=loss_function)

    wandb.watch(model, log="all")

    for epoch in range(1, config.general['epochs'] + 1):
        t = time.time()
        closs = runner.train(train_loader=train_loader,
                             epoch=epoch,
                             log_interval=config.general['log_interval']
                             )
        wandb.log({"loss": closs / (len(train_loader.dataset) / config.data['batch_size'])})

        wandb.log({"epoch_duration": (time.time() - t)})

        if epoch % config.general['model_save_interval'] == 0:  # test and save model every n epochs
            t = time.time()
            vloss = runner.test(test_loader=test_loader, log_interval=config.general['log_interval'])
            # wandb.log({"validation loss": vloss / (len(test_loader.dataset) / config.data['batch_size'])})
            wandb.log({"eval_duration": (time.time() - t)})
            model_path = wandb.run.dir + '/did_model_epoch_' + str(epoch) + '.pt'
            print("Saving model to " + model_path)
            torch.save(model.state_dict(), model_path)

        scheduler.step()
        wandb.log({"learning rate": optimizer.param_groups[0]['lr']})

    print('Finished Training')
