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
    set_random_seed()

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
                                         output_dim=args['output_dim'])
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
    test_l1_loss = run_an_eval_epoch(args, model, test_loader)
    print('test l1 loss {:.4f}'.format(test_l1_loss))

if __name__ == "__main__":
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Molecule Regression')
    parser.add_argument('-m', '--model', type=str, choices=['MPNN', 'SCHNET', 'MGCN'],
                        help='Model to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['Alchemy', 'ESOL'],
                        help='Dataset to use')
    parser.add_argument('-s', '--score', type=str, choices=['l1', 'RMSE'],
                        default='l1', help='Score function for evaluation.')
    args = parser.parse_args().__dict__
    args['exp'] = '_'.join([args['model'], args['dataset']])
    args.update(get_exp_configure(args['exp']))

    main(args)
