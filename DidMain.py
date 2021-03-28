import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import wandb
from DidDataset import DidDataset
from DidModel import DidModel
from DidModelRunner import DidModelRunner


def print_Config():
    print('data:')
    print("  train_data: " + config.data['train_dataset'])
    print("  test_data: " + config.data['test_dataset'])
    print("  batch_size: " + str(config.data['batch_size']) + ", shuffle: " + str(config.data['shuffle']))
    print('model:')
    print("  location: " + config.model['model_location'])
    print("  num_classes: " + str(config.model['num_classes']) + ", freeze_fairseq: " + str(
        config.model['freeze_fairseq']))
    print('general:')
    print("  num_workers: " + str(config.general['num_workers']))
    print("  epochs: " + str(config.general['epochs']))
    print("  optimizer: " + config.general['optimizer'])
    print("  loss_function: " + config.general['loss_function'])
    print("  log_interval: " + str(config.general['log_interval']) + ", model_save_interval: " + str(
        config.general['model_save_interval']))


def main(device_count, epochs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    args.world_size = device_count * args.nodes
    args.gpus = device_count
    args.epochs = epochs
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = '8711'  #
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    torch.dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)

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
    print("Test set size: " + str(len(test_set)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=args.world_size,
        rank=rank
    )

    # build data loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config.data['batch_size'],
                                               shuffle=False,
                                               sampler=train_sampler
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config.data['batch_size'],
                                              shuffle=False,
                                              **kwargs)

    # Loss Function and fitting exponential normalizing function
    if config.general['loss_function'] == 'nllLoss':
        loss_function = F.nll_loss.cuda(gpu)
        exp_norm_func = F.log_softmax
    else:
        raise SystemExit("you must specify loss_function for " + config.general['loss_function'])

    # Using more than one GPU
    if torch.cuda.device_count() > 1:
        print("Wrapping loss_function with DataParallelCriterion")
        loss_function = loss_function

    # create our own model with classifier on top of fairseq's xlsr_53_56k.pt
    model = DidModel(model_path=config.model['model_location'],
                     num_classes=config.model['num_classes'],
                     exp_norm_func=exp_norm_func,
                     freeze_fairseq=config.model['freeze_fairseq'])

    # Using more than one GPU
    if torch.cuda.device_count() > 1:
        model.cuda(gpu)
        model = DistributedDataParallel(model, device_ids=[gpu])

    # Optimizer
    print('optimizer_params:')
    if config.general['optimizer'] == 'adam':
        print('  lr: ' + str(config.optimizers[config.general['optimizer']]['lr']) + ', weight_decay: ' + str(
            config.optimizers[config.general['optimizer']]['weight_decay']))
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

    wandb.watch(model)

    for epoch in range(config.general['epochs']):
        closs = runner.train(train_loader=train_loader,
                             epoch=epoch,
                             log_interval=config.general['log_interval'])
        wandb.log({"loss": closs / (len(train_loader.dataset) / config.batch_size)})

        if epoch % config.general['model_save_interval'] == 0:  # test and save model every n epochs
            accuracy = runner.test(test_loader=test_loader)
            wandb.log({"accuracy": accuracy})
        #     model_path = './models/did_model_epoch_' + str(epoch) + '.pt'
        #     print("Saving model to " + model_path)
        #     torch.save(model.state_dict(), model_path)

        scheduler.step()

    print('Finished Training')



if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        did_config = json.load(f)

    # get device on which training should run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a new wandb run
    wandb.init(project='w2v_did', config=did_config, entity='ba-reisedomfiviapas',
               name=datetime.now().strftime("w2v_did " + "_%Y%m%d-%H%M%S"))
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    print_Config()

    # Using more than one GPU
    if torch.cuda.device_count() > 1:
        device_count = torch.cuda.device_count()
        print("Using:", device_count, "GPUs!")
        main(device_count=device_count, epochs=config.general['epochs'])
        # config_defaults["batch_size"] = config_defaults["batch_size"] * device_count
        # print("Multiplying batch * GPUs new batch_size=", config_defaults["batch_size"])

    # # define params for data loaders
    # kwargs = {'num_workers': config.general['num_workers'],
    #           'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu
    #
    # # build train data
    # csv_path_train = config.data['train_dataset'] + 'metadata.csv'  # file_path_train = './data/dev/segmented/'
    # train_set = DidDataset(csv_path_train, config.data['train_dataset'])
    # print("Train set size: " + str(len(train_set)))
    #
    # # build test data
    # csv_path_test = config.data['test_dataset'] + 'metadata.csv'  # file_path_test = './data/dev/segmented/'
    # test_set = DidDataset(csv_path_test, config.data['test_dataset'])
    # print("Test set size: " + str(len(test_set)))
    #
    # # build data loaders
    # train_loader = torch.utils.data.DataLoader(train_set,
    #                                            batch_size=config.data['batch_size'],
    #                                            shuffle=config.data['shuffle'],
    #                                            **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_set,
    #                                           batch_size=config.data['batch_size'],
    #                                           shuffle=config.data['shuffle'],
    #                                           **kwargs)
    #
    # # Loss Function and fitting exponential normalizing function
    # if config.general['loss_function'] == 'nllLoss':
    #     loss_function = F.nll_loss
    #     exp_norm_func = F.log_softmax
    # else:
    #     raise SystemExit("you must specify loss_function for " + config.general['loss_function'])
    #
    # # Using more than one GPU
    # if torch.cuda.device_count() > 1:
    #     print("Wrapping loss_function with DataParallelCriterion")
    #     loss_function = loss_function
    #
    # # create our own model with classifier on top of fairseq's xlsr_53_56k.pt
    # model = DidModel(model_path=config.model['model_location'],
    #                  num_classes=config.model['num_classes'],
    #                  exp_norm_func=exp_norm_func,
    #                  freeze_fairseq=config.model['freeze_fairseq'])
    #
    # # Using more than one GPU
    # if torch.cuda.device_count() > 1:
    #     print("Wrapping model with DataParallel")
    #     model = DataParallel(model)
    #
    # # Optimizer
    # print('optimizer_params:')
    # if config.general['optimizer'] == 'adam':
    #     print('  lr: ' + str(config.optimizers[config.general['optimizer']]['lr']) + ', weight_decay: ' + str(
    #         config.optimizers[config.general['optimizer']]['weight_decay']))
    #     optimizer = optim.Adam(model.parameters(),
    #                            lr=config.optimizers[config.general['optimizer']]['lr'],
    #                            weight_decay=config.optimizers[config.general['optimizer']]['weight_decay'])
    # else:
    #     raise SystemExit("you must specify optimizer for " + config.general['optimizer'])
    #
    # # Scheduler
    # print('scheduler_params:')
    # print('  step_size: ' + str(config.scheduler['step_size']) + ', gamma: ' + str(config.scheduler['gamma']))
    # scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                       step_size=config.scheduler['step_size'],
    #                                       gamma=config.scheduler['gamma'])
    #
    # # create runner for training and testing
    # runner = DidModelRunner(device=device,
    #                         model=model,
    #                         optimizer=optimizer,
    #                         scheduler=scheduler,
    #                         wandb=wandb,
    #                         loss_function=loss_function)
    #
    # wandb.watch(model)
    #
    # for epoch in range(config.general['epochs']):
    #     closs = runner.train(train_loader=train_loader,
    #                          epoch=epoch,
    #                          log_interval=config.general['log_interval'])
    #     wandb.log({"loss": closs / (len(train_loader.dataset) / config.batch_size)})
    #
    #     if epoch % config.general['model_save_interval'] == 0:  # test and save model every n epochs
    #         accuracy = runner.test(test_loader=test_loader)
    #         wandb.log({"accuracy": accuracy})
    #     #     model_path = './models/did_model_epoch_' + str(epoch) + '.pt'
    #     #     print("Saving model to " + model_path)
    #     #     torch.save(model.state_dict(), model_path)
    #
    #     scheduler.step()
    #
    # print('Finished Training')
