'''
main function of SRNN
'''
# https://discuss.pytorch.org/t/large-performance-gap-between-pytorch-and-keras-for-imdb-sentiment-analysis-model/135659

# tf dataset+ glove  'can not train successfully'
import sys

import mlflow
import numpy as np
import torch
# from metrics_helper import Score
from torch import nn, optim
from tqdm import tqdm

# from config import args as srnn_args
import configparser
from data_loader_util import load_data_imdb_glove, ag_news_load_data_by_glove, yelp_load_data_by_glove
from snn_models import SlayerSNN, SPASRNN, SPASNN
from ann_models import LSTM
import slayerSNN as slsnn
import torchmetrics
import os

home_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(home_dir, 'model/')
con_dir = os.path.join(home_dir, 'config.ini')

con = configparser.ConfigParser()
con.read(con_dir, encoding='utf-8')
opt = dict(dict(con.items('paths')), **dict(con.items("para")))

netParams = slsnn.params(home_dir + '/slayer_network.yaml')

best_acc = 0
best_f1 = 0
out_size = 0

# use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.manual_seed(int(opt['seed']))
np.random.seed(int(opt['seed']))

def main():
    global best_acc, best_f1, out_size
    test_dataloader, train_dataloader, glove_matrix = \
        dataset_builder(opt['dataset'])

    out_size = get_out_size(opt['dataset'])

    if opt['model_name'] == 'd2s_sRnn':
        resume_id = opt['dataset'] + '_' + opt['model_name'] + '_' + opt['recurrent_mode'] + \
            '_' + opt['reset_m'] + '_' + opt['readout_mode']
    elif opt['model_name'] == 'd2s_snn':
        resume_id = opt['dataset'] + '_' + opt['model_name'] + '_' + \
            opt['reset_m'] + '_' + opt['readout_mode']
    elif opt['model_name'] in ('slayer', 'lstm'):
        resume_id = opt['dataset'] + '_' + opt['model_name']
    else:
        raise Exception('The model you chose has not been implemented, \
                        please select from config.py')
    if opt['model_name'] == 'd2s_sRnn':
        print('LIF reset mode is {}'.format(opt['reset_m']))
        model = SPASRNN(glove_matrix, hid_dim=int(opt['nb_hidden']),
                        out_dim=out_size, dropout_prob=float(opt['dropout_prob']))
    if opt['model_name'] == 'd2s_snn':
        print('LIF reset mode is {}'.format(opt['reset_m']))
        model = SPASNN(glove_matrix, hid_dim=int(opt['nb_hidden']),
                       out_dim=out_size, dropout_prob=float(opt['dropout_prob']))
    if opt['model_name'] == 'slayer':
        model = SlayerSNN(glove_matrix, hid_dim=int(opt['nb_hidden']),
                          out_dim=out_size, dropout_prob=float(opt['dropout_prob']))
    if opt['model_name'] == 'lstm':
        model = LSTM(glove_matrix, int(opt['nb_embedding']), int(opt['nb_hidden']),
                     num_layers=1, out_dim=out_size)

    model.to(device=opt['device'])
    # loss func
    if opt['model_name'] in ('d2s_sRnn', 'd2s_snn', 'lstm'):
        # nn.BCELoss()# torch.nn.CrossEntropyLoss()
        loss_function = nn.CrossEntropyLoss().to(opt['device'])
    elif opt['model_name'] == 'slayer':
        loss_function = slsnn.loss(netParams).to(opt['device'])
    else:
        raise Exception('The model you chose has not been implemented, \
                        please select from config.py')

    # # slayer learning stat
    # if opt['model_name'] == 'slayer':
    #     stats = learningStats()

    # torchmetrics
    if out_size == 2:
        task_type = 'binary'
    elif out_size > 2:
        task_type = 'multiclass'
    acc_metric = torchmetrics.Accuracy(task=task_type,
                                       num_classes=out_size).to(opt['device'])
    f1_metric = torchmetrics.F1Score(task=task_type,
                                     num_classes=out_size).to(opt['device'])
    # recall_metric = torchmetrics.Recall(task=task_type,
    #                                     num_classes=out_size).to(opt['device'])
    # precision_metric = torchmetrics.Precision(task=task_type,
    #                                           num_classes=out_size).to(opt['device'])
    # auc_metric = torchmetrics.AUROC(task=task_type, \
    #               num_classes=out_size).to(device)

    if opt['opt'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), betas=(0.7, 0.995),
                               lr=float(opt['lr']))
    elif opt['opt'] == 'SDG': # TODO: need to check the format!
        optimizer = optim.SDG(model.parameters(), betas=(0.7, 0.995),
                              lr=float(opt['lr']))
    else:
        raise Exception('Please check the \'opt\' argument in config.py, for now, only support Adam and SDG.')

    for epoch in range(int(opt['epochs'])):
        epoch_train_losses = []
        epoch_train_accs = []
        epoch_val_losses = []

        # training
        model.train()
        t = tqdm(train_dataloader, desc='training...', file=sys.stdout)
        for labels, sentences in t:
            model.zero_grad()
            sentences = sentences.to(opt['device'])
            labels_ori = labels

            # slayer label adjust
            if opt['model_name'] == 'slayer':
                ori_labels_size = labels.size()[0]
                class_num = out_size
                labels_slayer = torch.zeros(
                    ori_labels_size, class_num, 1, 1, 1)
                for i in range(ori_labels_size):
                    labels_slayer[i][labels_ori[i].data] = 1
                labels = labels_slayer

            labels = labels.to(opt['device'])

            if opt['model_name'] == 'slayer':
                score = model(sentences)
            else:
                score = model(sentences).squeeze(1)

            # loss select based on model
            if opt['model_name'] in ('d2s_sRnn', 'd2s_snn', 'lstm'):
                loss = loss_function(score, torch.squeeze(
                    labels, dim=-1)).to(opt['device'])
            elif opt['model_name'] == 'slayer':
                # according to each batch length, modify the time length
                loss_function.errorDescriptor['tgtSpikeRegion']['stop'] = \
                    sentences.size()[1]
                loss = loss_function.numSpikes(
                    score, labels).to(opt['device'])
            else:
                raise Exception('The model you chose has not been implemented, \
                                please select from config.py')
            if opt['model_name'] == 'slayer':
                pred = slsnn.predict.getClass(score).to(opt['device'])
            else:
                # torch.nn.functional.sigmoid(score))
                # torch.argmax(torch.nn.functional.sigmoid(score).\
                #               float(), 1).float())
                # torch.nn.fun
                pred = torch.argmax(torch.softmax(score, dim=1), dim=1)
            train_batch_acc = acc_metric(pred, labels_ori)
            _ = f1_metric(pred, labels_ori)
            # train_batch_auc = auc_metric.update(pred, labels_ori)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(train_batch_acc.item())

            t.set_postfix(train_acc=format(np.mean(epoch_train_accs), '.4f'))

        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_acc = acc_metric.compute().to('cpu')
        epoch_train_f1 = f1_metric.compute().to('cpu')
        # epoch_train_auc = auc_metric.compute().to('cpu')

        # val_loss: {epoch_val_loss:.4f} val_acc: {epoch_val_acc:.4f}')
        print(
            f'epoch: {epoch + 1}/{0} train_loss: {epoch_train_loss:.4f} \
            train_acc: {epoch_train_acc:.4f} train_f1:{epoch_train_f1:.4f}'.format(int(opt['epochs'])))

        # metrics reset
        acc_metric.reset()
        f1_metric.reset()
        # recall_metric.reset()
        # precision_metric.reset()
        # auc_metric.reset()

        #  validation
        model.eval()
        with torch.no_grad():
            tm = tqdm(test_dataloader, desc='evaluating...', file=sys.stdout)
            for labels, sentences in tm:
                sentences = sentences.to(opt['device'])
                labels_ori = labels

                # slayer label adjust
                if opt['model_name'] == 'slayer':
                    ori_labels_size = labels.size()[0]
                    class_num = out_size
                    labels_slayer = \
                        torch.zeros(ori_labels_size, class_num, 1, 1, 1)
                    for i in range(ori_labels_size):
                        labels_slayer[i][labels_ori[i].data] = 1
                    labels = labels_slayer
                labels = labels.to(opt['device'])
                if opt['model_name'] == 'slayer':
                    score = model(sentences)
                else:
                    score = model(sentences).squeeze(1)
                # loss select based on model
                if opt['model_name'] in ('d2s_sRnn', 'd2s_snn', 'lstm'):
                    loss = loss_function(score, torch.squeeze(labels,
                                                              dim=-1)).to(opt['device'])
                elif opt['model_name'] == 'slayer':
                    # according to each batch length, modify the time length
                    loss_function.errorDescriptor['tgtSpikeRegion']['stop'] = \
                        sentences.size()[1]
                    loss = loss_function.numSpikes(
                        score, labels).to(opt['device'])
                else:
                    raise Exception('The model you chose has not been \
                                    implemented,please select from config.py')

                if opt['model_name'] == 'slayer':
                    pred = slsnn.predict.getClass(score).to(opt['device'])
                else:
                    pred = torch.argmax(torch.softmax(score, dim=1), dim=1)
                _ = acc_metric(pred, labels_ori)
                _ = f1_metric(pred, labels_ori)
                # test_batch_auc = auc_metric.update(pred, labels_ori)

                epoch_val_losses.append(loss.item())
                # epoch_val_accs.append(acc.item())
                # epoch_val_f1s.append(f1.item())
                # epoch_val_precisions.append(precision.item())
                # epoch_val_recalls.append(recall.item())

        epoch_val_loss = np.mean(epoch_val_losses)

        epoch_val_acc = acc_metric.compute().to('cpu')
        epoch_val_f1 = f1_metric.compute().to('cpu')
        # epoch_val_auc = auc_metric.compute().to('cpu')

        print(
            f'epoch: {epoch + 1}/{0} train_loss: {epoch_train_loss:.4f} \
            train_acc: {epoch_train_acc:.4f} train_f1:{epoch_train_f1:.4f} \
            val_loss: {epoch_val_loss:.4f} val_acc: {epoch_val_acc:.4f} \
            val_f1:{epoch_val_f1:.4f}'.format(int(opt['epochs'])))

        # metrics reset
        acc_metric.reset()
        f1_metric.reset()
        # recall_metric.reset()
        # precision_metric.reset()
        # auc_metric.reset()

        # Save the Best Model
        if out_size == 2:
            if epoch_val_acc >= best_acc:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                print('Save the Best_acc Model')
                state_best = {
                    'model': model.state_dict(),
                    'acc': epoch_val_acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_best, model_dir + resume_id + '.pt')
            best_acc = max(epoch_val_acc, best_acc)
        else:
            if epoch_val_f1 >= best_f1:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                print('Save the Best_f1 Model')
                state_best = {
                    'model': model.state_dict(),
                    'f1': epoch_val_f1,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_best, model_dir + resume_id + '.pt')
            best_f1 = max(epoch_val_f1, best_f1)

        # logging
        mlflow.log_metric('epoch', epoch)
        mlflow.log_metric('epoch_train_loss', epoch_train_loss, step=epoch)
        mlflow.log_metric('epoch_train_acc', epoch_train_acc, step=epoch)
        mlflow.log_metric('epoch_train_f1', epoch_train_f1, step=epoch)
        mlflow.log_metric('epoch_val_loss', epoch_val_loss, step=epoch)
        mlflow.log_metric('epoch_val_acc', epoch_val_acc, step=epoch)
        mlflow.log_metric('epoch_val_f1', epoch_val_f1, step=epoch)

        if opt['model_name'] == 'd2s_sRnn':
            weight_r = model.get_dweight_r()
            weight_r_min = weight_r.min()
            weight_r_max = weight_r.max()
            mlflow.log_metric('weight_R_min', weight_r_min,
                              step=epoch)  # logging
            mlflow.log_metric('weight_R_max', weight_r_max,
                              step=epoch)  # logging
            print(f'weight_R_min: {weight_r_min:.4f} \
                    weight_R_max: {weight_r_max:.4f}')

def get_out_size(dataset_name):
    if dataset_name == 'agnews':
        out_size_num = 4
    elif dataset_name == 'imdb':
        out_size_num = 2
    elif dataset_name == 'yelp':
        out_size_num = 5
    else:
        raise Exception('Please check the dataset you use, and set the out size in this function.')
    
    return out_size_num

def dataset_builder(dataset):
    if dataset == 'agnews':
        train_dataloader, test_dataloader, _, glove_matrix = \
            ag_news_load_data_by_glove(batch_size=int(opt['b']))
    elif dataset == 'imdb':
        train_dataloader, test_dataloader, _, glove_matrix = \
            load_data_imdb_glove(batch_size=int(opt['b']))
    elif dataset == 'yelp':
        train_dataloader, test_dataloader, _, glove_matrix = \
            yelp_load_data_by_glove(batch_size=int(opt['b']))
    else:
        raise Exception('The {0} dataset is not supported for now.'.format(dataset))
    return test_dataloader, train_dataloader, glove_matrix


if __name__ == '__main__':
    # hyper_params1_lr=[2e-4] # learning rates agnews:2e-4 imdb:2e-3
    # hyper_params2_bs = [32] # batch sizes agnews:32 imdb:128
    # #hyper_params1 = [2e-3]
    # #maxlens=[500,1000,1100,1200,1500,2000]
    # for param1 in hyper_params1_lr:
    #     learning_rate=param1
    #     for param2 in hyper_params2_bs:
    if opt['logger_name'] == 'your_name':
        raise Exception(
            'Please change \'logger_name\' to your ID in config.ini')
    os.environ['LOGNAME'] = opt['logger_name']
    mlflow.end_run()
    mlflow.set_tracking_uri(opt['tracking_url'])
    print('mlflow server setting successful')
    mlflow.set_experiment(opt['dataset'])  # agnews SRNN_test
    log_params_dict = {  # 'maxlen': maxlen,
        'embedding_dim': int(opt['nb_embedding']),
        'hidden_dim': int(opt['nb_hidden']),
        'learning_rate': float(opt['lr']),
        'batch_size': int(opt['b']),
        'n_epochs': int(opt['epochs']),
        'dropout': float(opt['dropout_prob']),
        'optimizer': opt['opt'],
        'dataset_name': opt['dataset'],
        'reset_mode': opt['reset_m'],
        'recurrent_mode': opt['recurrent_mode'],
        'readout_mode': opt['readout_mode']}
    with mlflow.start_run(run_name=opt['model_name']):
        mlflow.log_params(log_params_dict)
        main()
