import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import opt.opts as opts
import pickle
from torch.utils.tensorboard import SummaryWriter

import time
import logging

from model.model import losses, Audio2Face

def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

def save_checkpoint_new(checkpoint_save_path, model, optimizer, save_id):
    # if checkpoint_path doesn't exist
    if not os.path.isdir(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)
    model_path = os.path.join(checkpoint_save_path,
                                   'model_' + opt.id + '_' + format(int(save_id)) + '.pth')
    torch.save(model.state_dict(), model_path)
    print("model saved to {}".format(model_path))
    optimizer_path = os.path.join(checkpoint_save_path,
                                  'optimizer_' + opt.id + '_' + format(int(save_id)) + '.pth')
    torch.save(optimizer.state_dict(), optimizer_path)

def train(epochs,
          ckpt_epoch,
          model,
          checkpoint_save_path,
          loss_object,
          optimizer,
          scheduler,
          train_ds,
          test_ds,
          test_freq=5,
          save_freq=50,
          logger=None,
          writer=None):
    """Train the model
    Args:
        epochs: int, the number of epochs to train the model
        ckpt_epoch: int, the number of epochs to train the model
        model: tf.keras.Model, the model to train
        checkpoint_save_path: str, the path to save the checkpoint
        loss_object: tf.keras.losses, the loss function
        optimizer: tf.keras.optimizers, the optimizer
        train_ds: tf.data.Dataset, the training dataset
        test_ds: tf.data.Dataset, the test dataset
        test_freq: int, the frequency to test the model
        save_freq: int, the frequency to save the model
        logger: python logger
        writer: tensorboard writer
    """
    model = model.cuda()
    model.train()

    # Train the model for epochs
    for epoch in range(ckpt_epoch+1, epochs + 1):
        # Reset the metrics at the start of the next epoch
        torch.cuda.synchronize()
        time_start = time.time() # Record the start time of each epoch

        loss_sum = 0
        mse_sum = 0
        for train_data, labels in train_ds:
            predictions, emotion_input = model(train_data.cuda())
            loss, mse = loss_object(labels.cuda(), (predictions, emotion_input))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss_sum == 0:
                loss_sum = loss * train_data.shape[0]
                mse_sum = mse * train_data.shape[0]
            else:
                loss_sum += loss * train_data.shape[0]
                mse_sum += mse * train_data.shape[0]
        writer.add_scalar('Train/Loss', loss_sum.item(), epoch)
        torch.cuda.synchronize()
        time_end = time.time()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(
            # Log the training information
            f'Training Epoch-{epoch} '
            f'MSE: {mse_sum:.5f}, '
            f'Loss: {loss_sum:.5f}, '
            f'time {time_end - time_start:.2f}, '
            f'LR: {lr:.5f}, ')
        
        # Validate the model every test_freq epochs
        if (epoch % test_freq) == 0 or epoch == epochs:
            t_loss_sum = 0
            t_mse_sum = 0
            for test_data, test_labels in test_ds:
                with torch.no_grad():
                    predictions, emotion_input = model(test_data.cuda())
                    t_loss, t_mse = loss_object(test_labels.cuda(), (predictions, emotion_input))
                    if loss_sum == 0:
                        t_loss_sum = t_loss * test_data.shape[0]
                        t_mse_sum = t_mse * test_data.shape[0]
                    else:
                        t_loss_sum += t_loss * test_data.shape[0]
                        t_mse_sum += t_mse * test_data.shape[0]
            writer.add_scalar('Test/Loss', t_loss_sum.item(), epoch)
            logger.info(
                # Log the test information
                f'----- Test '
                f'Loss: {t_loss_sum:.5f} '
                f'MSE: {t_mse_sum:.5f}, ')

        # Save the model every save_freq epochs
        if (epoch % save_freq) == 0 or epoch == epochs:
            save_checkpoint_new(checkpoint_save_path, model, optimizer, epoch)
            logger.info(f"----- Save Checkpoint: {checkpoint_save_path}")



if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu) # Allow GPU memory growth

    # Set random seed
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataSet = opt.dataset

    # Training Parameters
    EPOCHS = opt.epoch    # The number of epochs to train the model
    CKPT_EPOCHS = 0 # The epoch to restore the model
    test_freq = 10  # Test the model every test_freq epochs
    save_freq = 10  # Save the model every save_freq epochs
    batch_size = opt.bs # Batch size
    initial_learning_rate = opt.lr   # Initial learning rate
    keep_pro = 0.5  # Dropout rate

    # Path
    project_dir = './'
    output_feature = opt.output_feature
    output_path = opt.output_path + '_' + opt.id + '_' + output_feature
    finetune = opt.finetune

    # create SummaryWriter
    writer = SummaryWriter(log_dir=output_path)

    checkpoint_save_path = os.path.join(output_path, 'checkpoint/Audio2Face')
    model_save_path = os.path.join(output_path,  'models/Audio2Face')

    # Create output folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data_dir = os.path.join(project_dir, dataSet)
    logger = get_logger(filename=os.path.join(output_path, 'log.txt'))

    # Load data
    x_train = np.load(os.path.join(data_dir, 'train_data.npy'))
    x_val = np.load(os.path.join(data_dir, 'val_data.npy'))
    if output_feature == 'mouth':
        y_train_path = 'train_label_var_mouth.npy'
        y_val = 'val_label_var_mouth.npy'
    elif output_feature == 'other':
        y_train_path = 'train_label_var_other.npy'
        y_val = 'val_label_var_other.npy'
    elif output_feature == 'head':
        y_train_path = 'train_label_var_head.npy'
        y_val = 'val_label_var_head.npy'
    y_train = np.load(os.path.join(data_dir, y_train_path))
    y_val = np.load(os.path.join(data_dir, y_val))

    print('train data len:', x_train.shape[0])
    output_size = y_val.shape[1]
    
    # Convert to tensor
    x_train = torch.as_tensor(x_train, dtype=torch.float32)[:-1]
    y_train = torch.as_tensor(y_train, dtype=torch.float32)[:-1]
    x_val = torch.as_tensor(x_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32)

    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    # Build model
    model = Audio2Face(output_size, keep_pro)

    if finetune:
        model_path = opt.model_path
        state_dict = torch.load(model_path)
        print('load model from: ', model_path)
        # get rid of fc layer
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('OutputLayer.output_layer.2')}
        print([k for k, v in state_dict.items()])

        model.load_state_dict(state_dict, strict=False)

    # Setting optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = x_train.shape[0] // batch_size * 10, gamma=opt.gamma)

    # Setting loss function
    loss_object = losses
    
    logger.info(f'\n\n--------------------------------------------------------------')
    # Satrt training
    train(EPOCHS, CKPT_EPOCHS, model, checkpoint_save_path, loss_object,
          optimizer, scheduler, train_loader, val_loader, test_freq, save_freq, logger, writer)
          
    # Train Finished
    logger.info(f'Train Finished')
    writer.close()