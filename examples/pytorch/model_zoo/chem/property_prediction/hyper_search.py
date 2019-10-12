import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dgl import model_zoo

from utils import Meter, set_random_seed, collate_molgraphs, EarlyStopping, \
    load_dataset_for_regression

def regress(args, model, bg):
    if args['model'] == 'MPNN':
        h = bg.ndata.pop('n_feat')
        e = bg.edata.pop('e_feat')
        h, e = h.to(args['device']), e.to(args['device'])
        return model(bg, h, e)
    else:
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_types, edge_distances = node_types.to(args['device']), \
                                     edge_distances.to(args['device'])
        return model(bg, node_types, edge_distances)

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = regress(args, model, bg)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * bg.batch_size
        train_meter.update(prediction, labels, masks)
    total_loss /= len(data_loader.dataset)
    if args['score'] == 'l1':
        total_score = train_meter.l1_loss_averaged_over_tasks() / len(data_loader.dataset)
    elif args['score'] == 'RMSE':
        total_score = train_meter.rmse_averaged_over_tasks() / len(data_loader.dataset)
    print('epoch {:d}/{:d}, training loss {:.4f}, training score {:.4f}'.format(
        epoch + 1, args['num_epochs'], total_loss, total_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            prediction = regress(args, model, bg)
            eval_meter.update(prediction, labels, masks)
    if args['score'] == 'l1':
        total_score = eval_meter.l1_loss_averaged_over_tasks() / len(data_loader.dataset)
    elif args['score'] == 'RMSE':
        total_score = eval_meter.rmse_averaged_over_tasks() / len(data_loader.dataset)
    return total_score

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(args['seed'])

    # Interchangeable with other datasets
    train_set, val_set, test_set = load_dataset_for_regression(args)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    if test_set is not None:
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)

    if args['model'] == 'MPNN':
        model = model_zoo.chem.MPNNModel(node_input_dim=args['node_in_feats'],
                                         edge_input_dim=args['edge_in_feats'],
                                         output_dim=args['output_dim'],
                                         node_hidden_dim=args['node_hidden_dim'],
                                         edge_hidden_dim=args['edge_hidden_dim'],
                                         num_step_message_passing=args['num_step_message_passing'],
                                         num_step_set2set=args['num_step_set2set'],
                                         num_layer_set2set=args['num_layer_set2set'])
    elif args['model'] == 'SCHNET':
        model = model_zoo.chem.SchNet(norm=args['norm'], output_dim=args['output_dim'])
        model.set_mean_std(train_set.mean, train_set.std, args['device'])
    elif args['model'] == 'MGCN':
        model = model_zoo.chem.MGCNModel(norm=args['norm'], output_dim=args['output_dim'])
        model.set_mean_std(train_set.mean, train_set.std, args['device'])
    model.to(args['device'])

    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    stopper = EarlyStopping(mode='lower', patience=args['patience'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation score {:.4f}, best validation score {:.4f}'.format(
            epoch + 1, args['num_epochs'], val_score, stopper.best_score))
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('test score {:.4f}'.format(test_score))
    return stopper.best_score, test_score

if __name__ == "__main__":
    import argparse
    import numpy as np
    from copy import deepcopy
    from pprint import pprint
    from sklearn.model_selection import ParameterGrid

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Regression')
    parser.add_argument('-m', '--model', type=str, choices=['MPNN', 'SCHNET', 'MGCN'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, default='ESOL')
    parser.add_argument('-s', '--score', type=str, default='RMSE')
    parser.add_argument('-nr', '--num-runs', type=int, default=5)
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    hyper_to_tune = {
        'lr': [0.001, 0.0001],
        'patience': [30, 60],
        'node_hidden_dim': [32, 64],
        'edge_hidden_dim': [64, 128],
        'num_step_message_passing': [3, 6],
        'num_step_set2set': [3, 6],
        'num_layer_set2set': [1, 3]
    }
    hyper_choices = list(ParameterGrid(hyper_to_tune))
    num_choices = len(hyper_choices)

    all_mean_val_scores = []
    all_mean_test_scores = []

    for i, setting in enumerate(hyper_choices):
        setting_val_scores = []
        setting_test_scores = []

        for run in range(args['num_runs']):
            print('Processing choice {:d}/{:d}, run {:d}/{:d}'.format(
                i + 1, num_choices, run + 1, args['num_runs']))
            seed = run * 1000
            setting_args = deepcopy(args)
            setting_args.update(setting)
            setting_args['seed'] = seed
            val_metric, test_metric = main(setting_args)
            setting_val_scores.append(val_metric)
            setting_test_scores.append(test_metric)

        all_mean_val_scores.append(np.mean(setting_val_scores))
        all_mean_test_scores.append(np.mean(setting_test_scores))

    best_setting_idx = np.argmax(all_mean_val_scores)
    best_val_score = all_mean_val_scores[best_setting_idx]
    best_test_score = all_mean_test_scores[best_setting_idx]
    summary = hyper_choices[best_setting_idx]
    summary.update({
        'best_val_score': best_val_score,
        'best_test_score': best_test_score
    })
    pprint(summary)
