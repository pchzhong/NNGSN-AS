import logging
import math

import numpy as np
import os
import random
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import datasets
import models
from utils.count_params import compute_num_params


class TrainUtils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def run(self, ):
        args = self.args
        save_dir = self.save_dir

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        fault_sets = dict()
        # Load the datasets
        data_set = getattr(datasets, args.data_name)(
            data_dir=args.data_dir, signal_norm=args.signal_norm)
        fault_sets['train'], fault_sets['valid'], fault_sets['test'] = \
            data_set.data_prepare(
                valid_ratio=args.valid_ratio, test_ratio=args.test_ratio, partial_train=args.partial_train,
                mimic_noise=args.mimic_noise, noise_seed=args.noise_seed,
                train_noise_ratio=args.train_noise_ratio, train_snr=args.train_snr,
                exclude_noise=args.exclude_noise)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        fault_loaders = {
            x: DataLoader(
                fault_sets[x], batch_size=args.batch_size,
                shuffle=(True if x == 'train' else False),
                pin_memory=(True if device == 'cuda' else False),
                num_workers=args.num_workers) for x in ['train', 'valid', 'test']}

        # Define the model
        model = getattr(models, args.model_name)(
            in_channel=data_set.num_in_chns, out_channel=data_set.num_classes)
        total_params, total_trainable_params = compute_num_params(model)
        logging.info(f"total-params-of-{model.__class__.__name__}: {total_params}, "
                     f"and {total_trainable_params} trainable")
        model.to(device)
        # Define the loss
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        # Define the learning rate decay
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.t_max, eta_min=args.eta_min)

        step = 1
        best_acc = 0.0
        best_path = 'fake.pth'
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        batch_train_loss = []
        batch_train_acc = []
        batch_valid_loss = []
        batch_valid_acc = []
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_valid_loss = []
        epoch_valid_acc = []

        train_start = time.time()

        for epoch_idx in range(args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch_idx + 1, args.max_epoch) + '-' * 5)

            # Each epoch has a training and val phase
            for phase in ['train', 'valid']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                for batch_idx, (inputs, labels) in enumerate(fault_loaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        logits = model(inputs)
                        loss = criterion(logits, labels.long())
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        if phase == 'train':
                            batch_train_loss.append(loss.item())
                            batch_train_acc.append(correct / inputs.size(0))
                        else:
                            batch_valid_loss.append(loss.item())
                            batch_valid_acc.append(correct / inputs.size(0))

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            lr_scheduler.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            num_batch = math.ceil(len(fault_loaders[phase].dataset) / args.batch_size)

                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step
                                sample_per_sec = 1.0 * batch_count / train_time
                                if num_batch == batch_idx + 1:
                                    logging.info(
                                        'Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f}, '
                                        '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                            epoch_idx + 1, len(fault_loaders[phase].dataset),
                                            len(fault_loaders[phase].dataset),
                                            batch_loss, batch_acc, sample_per_sec, batch_time))
                                else:
                                    logging.info(
                                        'Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f}, '
                                        '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                            epoch_idx + 1, (batch_idx + 1) * len(inputs),
                                            len(fault_loaders[phase].dataset),
                                            batch_loss, batch_acc, sample_per_sec, batch_time))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                epoch_loss = epoch_loss / len(fault_loaders[phase].dataset)
                epoch_acc = epoch_acc / len(fault_loaders[phase].dataset)

                if phase == 'train':
                    epoch_train_loss.append(epoch_loss)
                    epoch_train_acc.append(epoch_acc)
                else:
                    epoch_valid_loss.append(epoch_loss)
                    epoch_valid_acc.append(epoch_acc)

                logging.info(
                    'Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                        epoch_idx + 1, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start))

                # save the model
                if phase == 'valid' and epoch_acc > best_acc:

                    model_state_dic = model.state_dict()
                    # save the best model according to the val accuracy
                    if os.path.exists(best_path):
                        os.remove(best_path)
                    best_acc = epoch_acc
                    best_path = os.path.join(save_dir, f'{epoch_idx + 1}-{best_acc:.4f}-best_model.pth')
                    logging.info(
                        f"save best model epoch {epoch_idx + 1}, acc {epoch_acc:.4f}")
                    torch.save(model_state_dic, best_path)

        train_end = time.time()
        train_time = train_end - train_start
        logging.info(
            f'train-Time: {train_time:.2f}')

        np.savetxt(
            os.path.join(save_dir, 'batch_train_monitor.csv'),
            np.stack([torch.tensor(batch_train_loss).cpu().numpy(), torch.tensor(batch_train_acc).cpu().numpy()], axis=1),
            delimiter=',')

        np.savetxt(
            os.path.join(save_dir, 'batch_valid_monitor.csv'),
            np.stack([torch.tensor(batch_valid_loss).cpu().numpy(), torch.tensor(batch_valid_acc).cpu().numpy()], axis=1),
            delimiter=',')

        np.savetxt(
            os.path.join(save_dir, 'epoch_train_monitor.csv'),
            np.stack([torch.tensor(epoch_train_loss).cpu().numpy(), torch.tensor(epoch_train_acc).cpu().numpy()], axis = 1),
            delimiter=',')

        np.savetxt(
            os.path.join(save_dir, 'epoch_valid_monitor.csv'),
            np.stack([torch.tensor(epoch_valid_loss).cpu().numpy(), torch.tensor(epoch_valid_acc).cpu().numpy()], axis=1),
            delimiter=',')

        # test
        test_start = time.time()
        model.load_state_dict(torch.load(best_path))
        epoch_test_loss = 0.0
        epoch_test_acc = 0.0

        model.eval()

        for test_inputs, test_labels in fault_loaders['test']:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            logits = model(test_inputs)
            loss = criterion(logits, test_labels.long())
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, test_labels).float().sum().item()
            loss_temp = loss.item() * test_inputs.size(0)
            epoch_test_loss += loss_temp
            epoch_test_acc += correct
        epoch_test_loss = epoch_test_loss / len(fault_loaders['test'].dataset)
        epoch_test_acc = epoch_test_acc / len(fault_loaders['test'].dataset)
        test_end = time.time()
        test_time = test_end - test_start
        logging.info(
            f'test-Loss: {epoch_test_loss:.4f} test-Acc: {epoch_test_acc:.4f} test-Time: {test_time:.2f}')

