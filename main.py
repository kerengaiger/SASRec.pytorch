import os
import time
import torch
from ax.service.managed_loop import optimize
import argparse

from model import SASRec
from tqdm import tqdm
from utils import *
import pickle
import pathlib
from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def train_with_cnfg(cnfg):
    if not os.path.isdir(cnfg['dataset'] + '_' + cnfg['train_dir']):
        os.makedirs(cnfg['dataset'] + '_' + cnfg['train_dir'])
    with open(os.path.join(cnfg['dataset'] + '_' + cnfg['train_dir'], 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(cnfg.items(), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition(cnfg['dataset'], cnfg['split'])
    [user_train, user_valid, _, usernum, itemnum, _] = dataset

    if cnfg['is_final_train']:
        print('Final train! Merge between train and valid')
        for usr in user_train.keys():
            user_train[usr] += user_valid[usr]

    num_batch = len(user_train) // cnfg['batch_size'] # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(cnfg['dataset'] + '_' + cnfg['train_dir'], 'log.txt'), 'w')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=cnfg['batch_size'], maxlen=cnfg['maxlen'], n_workers=3)
    model = SASRec(usernum, itemnum, cnfg).to(cnfg['device']) # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train() # enable model training

    epoch_start_idx = 1
    if cnfg['state_dict_path'] != '':
        try:
            model.load_state_dict(torch.load(cnfg['state_dict_path'], map_location=torch.device(cnfg['device'])))
            tail = cnfg['state_dict_path'][cnfg['state_dict_path'].find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(cnfg['state_dict_path'])
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()

    if cnfg['inference_only']:
        model.eval()
        t_test = evaluate(model, dataset, cnfg)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=cnfg['lr'], betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    writer = SummaryWriter(log_dir=cnfg['log_dir'])

    for epoch in range(epoch_start_idx, cnfg['num_epochs'] + 1):
        if cnfg['inference_only']: break # just to decrease identition
        loss_epoch = 0.0
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=cnfg['device']), torch.zeros(neg_logits.shape,
                                                                                                      device=cnfg['device'])
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += cnfg['l2_emb'] * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            loss_epoch += loss.item()
        print("loss in epoch {}: {}".format(epoch, loss_epoch)) # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # t_test = evaluate(model, dataset, cnfg)
            t_valid = evaluate_valid(model, dataset, cnfg)
            writer.add_scalar("HR@20/valid", t_valid[1], epoch)
            t_test = ''
            print('epoch:%d, time: %f(s), valid (NDCG@20: %.4f, HR@20: %.4f)' % (epoch, T, t_valid[0], t_valid[1],))
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == cnfg['num_epochs']:
            folder = cnfg['dataset'] + '_' + cnfg['train_dir']
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(cnfg['num_epochs'], cnfg['lr'], cnfg['num_blocks'], cnfg['num_heads'],
                                 cnfg['hidden_units'], cnfg['maxlen'])
            torch.save(model.state_dict(), os.path.join(folder, fname))
            if cnfg['is_final_train']:
                ndcg_test, hr_test, preds_test = evaluate(model, dataset, cnfg)
            preds_test.to_csv(os.path.join(folder, 'preds_test.csv'), index=False, header=False)

    writer.flush()
    writer.close()
    f.close()
    sampler.close()
    print("Done config")
    return t_valid[1]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--split_char', default=',', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--trials', default=5, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default='', type=str)
parser.add_argument('--is_final_train', default=False, type=str2bool)
parser.add_argument('--cnfg_file', default='cnfg.pkl', type=str)
parser.add_argument('--log_dir', default='tensorboard/', type=str)

args = parser.parse_args()

if args.is_final_train:
    print('Train with loaded config')
    cnfg = pickle.load(open(os.path.join(args.dataset + '_' + args.train_dir, args.cnfg_file), 'rb'))
    train_with_cnfg(cnfg)

else:
    best_parameters, values, _experiment, _cur_model = optimize(
        parameters=[
            {"name": "dataset", "type": "fixed", "value_type": "str", "value": args.dataset},
            {"name": "train_dir", "type": "fixed", "value_type": "str", "value": args.train_dir},
            {"name": "batch_size", "type": "choice", "value_type": "int", "values": [32, 64, 128, 256]},
            {"name": "lr", "type": "fixed", "value_type": "float", "value": args.lr},
            {"name": "maxlen", "type": "choice", "value_type": "int", "values": [50, 100, 150, 200]},
            {"name": "hidden_units", "type": "choice", "value_type": "int", "values": [50, 60, 70, 80, 90, 100]},
            {"name": "num_blocks", "type": "choice", "value_type": "int", "values": [2, 3, 4]},
            {"name": "num_heads", "type": "choice", "value_type": "int", "values": [1, 2]},
            {"name": "num_epochs", "type": "fixed", "value_type": "int", "value": 201},
            {"name": "dropout_rate", "type": "range", "value_type": "float", "bounds": [0.1, 0.6]},
            {"name": "l2_emb", "type": "fixed", "value_type": "float", "value": args.l2_emb},
            {"name": "device", "type": "fixed", "value_type": "str", "value": args.device},
            {"name": "is_final_train", "type": "fixed", "value_type": "bool", "value": args.is_final_train},
            {"name": "inference_only", "type": "fixed", "value_type": "bool", "value": False},
            {"name": "split", "type": "fixed", "value_type": "str", "value": args.split_char},
            {"name": "state_dict_path", "type": "fixed", "value_type": "str", "value": args.state_dict_path},
            {"name": "log_dir", "type": "fixed", "value_type": "str", "value": args.log_dir},

        ],
        evaluation_function=train_with_cnfg,
        minimize=False,
        objective_name='hr10',
        total_trials=args.trials
    )
    pickle.dump(best_parameters, open(os.path.join(args.dataset + '_' + args.train_dir, 'cnfg.pkl'), "wb"))
    best_parameters['is_final_train'] = True
    train_with_cnfg(best_parameters)

