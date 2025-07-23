import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import yaml
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
import numpy as np

from dataset import genSpoof_list,Dataset_train,Dataset_eval
from utils import compute_eer, compute_eval_eer, compute_multi_accuracy
from model import W2V2_AASIST_Model

def get_last_epoch(model_save_path):
    """获取最后一个epoch的编号"""
    epoch_files = [f for f in os.listdir(model_save_path) if f.startswith('epoch_') and f.endswith('.pth')]
    if not epoch_files:
        return -1
    epochs = [int(f.split('_')[1].split('.')[0]) for f in epoch_files]
    return max(epochs)

class MultiTaskWeightedLoss(nn.Module):
    def __init__(self, model_type, is_learnable=False):
        super().__init__()
        self.model_type=model_type
        if model_type=="M1":
            self.weight = nn.Parameter(torch.tensor([0.25,0.25,0.25,0.25]), requires_grad= is_learnable)
        elif model_type=="M2":
            self.weight = nn.Parameter(torch.tensor([1/3,1/3,1/3]), requires_grad= is_learnable)
        else:
            self.weight = nn.Parameter(torch.tensor([0.5,0.5]), requires_grad= is_learnable)
            

        binary_weight = torch.FloatTensor([0.1, 0.9]).to(device)
        multi_weight_as = torch.FloatTensor([0.1, 0.3, 0.3, 0.3]).to(device)
        multi_weight_ds = torch.FloatTensor([0.1, 0.45, 0.45]).to(device)
        multi_weight_qs = torch.FloatTensor([0.1, 0.3, 0.3, 0.3]).to(device)
    
        self.binary_criterion = nn.CrossEntropyLoss(weight=binary_weight)
        self.multi_criterion_as = nn.CrossEntropyLoss(weight=multi_weight_as)
        self.multi_criterion_ds = nn.CrossEntropyLoss(weight=multi_weight_ds)
        self.multi_criterion_qs = nn.CrossEntropyLoss(weight=multi_weight_qs)

    def forward(self, outputs, labels):
        
        binary_loss = self.binary_criterion(outputs['binary'], labels['binary']) if self.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"] else None
        multi_loss_as = self.multi_criterion_as(outputs['as'], labels['as']) if self.model_type in ["S_AUX", "D_AUX", "M1", "M2"] else None
        multi_loss_ds = self.multi_criterion_ds(outputs['ds'], labels['ds']) if self.model_type in ["S_DEC", "D_DEC", "M1", "M2"] else None
        multi_loss_qs = self.multi_criterion_qs(outputs['qs'], labels['qs']) if self.model_type in ["S_VQ", "D_VQ", "M1", "M2"] else None
        
        W=torch.softmax(self.weight,dim=0)
        
        if self.model_type=="M1":
            _loss = W[0]*binary_loss+W[1]*multi_loss_as+W[2]*multi_loss_ds+W[3]*multi_loss_qs
        elif self.model_type=="M2":
            _loss = W[0]*multi_loss_as+W[1]*multi_loss_ds+W[2]*multi_loss_qs
        elif self.model_type=="D_AUX":
            _loss = W[0]*binary_loss+W[1]*multi_loss_as
        elif self.model_type=="D_DEC":
            _loss = W[0]*binary_loss+W[1]*multi_loss_ds
        elif self.model_type=="D_VQ":
            _loss = W[0]*binary_loss+W[1]*multi_loss_qs
        elif self.model_type=="S_BIN":
            _loss = binary_loss
        elif self.model_type=="S_AUX":
            _loss = multi_loss_as
        elif self.model_type=="S_DEC":
            _loss = multi_loss_ds
        elif self.model_type=="S_VQ":
            _loss = multi_loss_qs
        
        return _loss,(binary_loss,multi_loss_as,multi_loss_ds,multi_loss_qs)

def evaluate_metrics(dev_loader, model, device,loss_func):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    loss_func.eval()
    
    all_binary_scores = []
    all_binary_labels = []
    
    all_multi_scores_as = []
    all_multi_labels_as = []

    all_multi_scores_ds = []
    all_multi_labels_ds = []

    all_multi_scores_qs = []
    all_multi_labels_qs = []

    
    progress_bar = tqdm(dev_loader, desc='Validating')
    for batch_x, batch_y_binary, batch_y_multi_as,batch_y_multi_ds,batch_y_multi_qs in progress_bar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y_binary = batch_y_binary.view(-1).type(torch.int64).to(device)
        batch_y_multi_as = batch_y_multi_as.view(-1).type(torch.int64).to(device)
        batch_y_multi_ds = batch_y_multi_ds.view(-1).type(torch.int64).to(device)
        batch_y_multi_qs = batch_y_multi_qs.view(-1).type(torch.int64).to(device)
        labels={"binary":batch_y_binary,"as":batch_y_multi_as,"ds":batch_y_multi_ds,"qs":batch_y_multi_qs}
        
        with torch.no_grad():
            outputs = model(batch_x)
            batch_loss,_=loss_func(outputs,labels)
            
            val_loss += (batch_loss.item() * batch_size)

            if model.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"]:
                binary_scores = outputs["binary"].softmax(dim=1)[:, 1].cpu().numpy()
                all_binary_scores.extend(binary_scores)
                all_binary_labels.extend(batch_y_binary.cpu().numpy())

            if model.model_type in ["S_AUX", "D_AUX", "M1", "M2"]:
                multi_scores_as = outputs["as"].softmax(dim=1).cpu().numpy()
                all_multi_scores_as.extend(multi_scores_as)
                all_multi_labels_as.extend(batch_y_multi_as.cpu().numpy())

            if model.model_type in ["S_DEC", "D_DEC", "M1", "M2"]:
                multi_scores_ds = outputs["ds"].softmax(dim=1).cpu().numpy()
                all_multi_scores_ds.extend(multi_scores_ds)
                all_multi_labels_ds.extend(batch_y_multi_ds.cpu().numpy())

            if model.model_type in ["S_VQ", "D_VQ", "M1", "M2"]:
                multi_scores_qs = outputs["qs"].softmax(dim=1).cpu().numpy()
                all_multi_scores_qs.extend(multi_scores_qs)
                all_multi_labels_qs.extend(batch_y_multi_qs.cpu().numpy())
        
        progress_bar.set_postfix({'Loss': f'{val_loss/num_total:.4f}'})
    
    val_loss /= num_total

    eer,as_acc,ds_acc,qs_acc=None,None,None,None
    
    # compute binary backend EER
    if model.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"]:
        binary_scores = np.array(all_binary_scores)
        binary_labels = np.array(all_binary_labels)
        target_scores = binary_scores[binary_labels == 1]
        nontarget_scores = binary_scores[binary_labels == 0]
        eer, _ = compute_eer(target_scores, nontarget_scores)
    
    # compute source tracing backend F1 scores
    if model.model_type in ["S_AUX", "D_AUX", "M1", "M2"]:
        multi_pred_as = np.argmax(all_multi_scores_as, axis=1)
        as_acc = np.mean(multi_pred_as == all_multi_labels_as)

    if model.model_type in ["S_DEC", "D_DEC", "M1", "M2"]:
        multi_pred_ds = np.argmax(all_multi_scores_ds, axis=1)
        ds_acc = np.mean(multi_pred_ds == all_multi_labels_ds)

    if model.model_type in ["S_VQ", "D_VQ", "M1", "M2"]:
        multi_pred_qs = np.argmax(all_multi_scores_qs, axis=1)
        qs_acc = np.mean(multi_pred_qs == all_multi_labels_qs)
        
    return val_loss, (eer, as_acc,ds_acc,qs_acc)

def produce_evaluation_file(dataset, model, model_save_path, epoch):
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    model.eval()
    device=model.device
    model_type = model.model_type
    
    save_dir=model_save_path
          
    trash_file=os.path.join(save_dir, "trash.txt")

    bin_sv_path=os.path.join(save_dir, f"bin_score_epoch_{epoch}.txt") if model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"] else trash_file
    aux_sv_path=os.path.join(save_dir, f"aux_score_epoch_{epoch}.txt") if model_type in ["S_AUX", "D_AUX", "M1", "M2"] else trash_file
    dec_sv_path=os.path.join(save_dir, f"dec_score_epoch_{epoch}.txt") if model_type in ["S_DEC", "D_DEC", "M1", "M2"] else trash_file
    vq_sv_path=os.path.join(save_dir,  f" vq_score_epoch_{epoch}.txt") if model_type in ["S_VQ", "D_VQ", "M1", "M2"] else trash_file
    
    progress_bar = tqdm(data_loader, desc='Evaluating')
    with open(bin_sv_path, 'w') as f_bin, open(aux_sv_path, 'w') as f_aux, open(dec_sv_path, 'w') as f_dec, open(vq_sv_path, 'w') as f_vq:
        for batch_x, utt_id in progress_bar:
            batch_x = batch_x.to(device)
    
            with torch.no_grad():
                outputs = model(batch_x)

                binary_scores = softmax(outputs['binary'], dim=1)[:, 1].cpu().numpy() if model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"] else None
                as_scores = softmax(outputs['as'], dim=1).cpu().numpy() if model_type in ["S_AUX", "D_AUX", "M1", "M2"] else None
                ds_scores = softmax(outputs['ds'], dim=1).cpu().numpy() if model_type in ["S_DEC", "D_DEC", "M1", "M2"] else None
                qs_scores = softmax(outputs['qs'], dim=1).cpu().numpy() if model_type in ["S_VQ", "D_VQ", "M1", "M2"] else None

                if binary_scores is not None:
                    for id, s in zip(utt_id, binary_scores):
                        f_bin.write('{} {}\n'.format(id, s))
                if as_scores is not None:
                    for id, s in zip(utt_id, as_scores):
                        f_aux.write('{} {}\n'.format(id, " ".join(map(str, s))))
                if ds_scores is not None:
                    for id, s in zip(utt_id, ds_scores):
                        f_dec.write('{} {}\n'.format(id, " ".join(map(str, s))))
                if qs_scores is not None:
                    for id, s in zip(utt_id, qs_scores):
                        f_vq.write('{} {}\n'.format(id, " ".join(map(str, s))))

    if os.path.exists(trash_file):
        os.remove(trash_file)
    print('Scores saved to {}'.format(save_dir))

def train_epoch(train_loader, model, args, optim, device,loss_func):
    lr=args.lr
    running_loss = 0
    num_total = 0.0
    model.train()
    loss_func.train()
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_x, batch_y_binary, batch_y_multi_as,batch_y_multi_ds,batch_y_multi_qs in progress_bar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y_binary = batch_y_binary.view(-1).type(torch.int64).to(device)
        batch_y_multi_as = batch_y_multi_as.view(-1).type(torch.int64).to(device)
        batch_y_multi_ds = batch_y_multi_ds.view(-1).type(torch.int64).to(device)
        batch_y_multi_qs = batch_y_multi_qs.view(-1).type(torch.int64).to(device)

        labels={"binary":batch_y_binary,"as":batch_y_multi_as,"ds":batch_y_multi_ds,"qs":batch_y_multi_qs}
        
        outputs = model(batch_x)
        
        batch_loss,_=loss_func(outputs,labels)
        
        running_loss += (batch_loss.item() * batch_size)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if args.use_learnable_weight and (not args.model_type.startswith("S")):
            debug_W=loss_func.weight.cpu().detach().numpy()
            exp_W = np.exp(debug_W - np.max(debug_W))  
            softmax_debug_W = exp_W / np.sum(exp_W)
            progress_bar.set_postfix({'Loss': f'{running_loss/num_total:.4f}','Weight of loss:':f"{softmax_debug_W}"})
        else:
            progress_bar.set_postfix({'Loss': f'{running_loss/num_total:.4f}'})
       
    running_loss /= num_total
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for CoRS dataset')
    parser.add_argument('--metadata_path', type=str, default='./metadata/training/', help='Change with path to user\'s LA database protocols directory address')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--save_dir', type=str,
                        default="models", help='Model checkpoint save directory')
    
    parser.add_argument("--model_type",
                            choices=[
                                "S_BIN", "S_VQ", "S_AUX", "S_DEC",
                                "D_VQ", "D_AUX", "D_DEC",
                                "M1", "M2"
                            ],
                            help=(
                                "Model type:\n"
                                "  S_{$task} : Single-task learning of {$task}\n"
                                "  D_{$task} : Dual-task learning of spoof detection + {$task}\n"
                                "  M1        : Multi-task learning of spoof detection + VQ + AUX + DEC tasks\n"
                                "  M2        : Multi-task learning of VQ + AUX + DEC tasks"
                            )
                        )
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--w2v2_pretrain_path', type=str, default="./xlsr2_300m.pt",
                        help='Path to xlsr pretrained weight')
    parser.add_argument('--use_learnable_weight', action='store_true', default=False,help='Use learnable weight or constant weight')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument("--sampling_strategy", default="AUX",
                            choices=["VQ", "AUX", "DEC"],
                            help=("Which taxonomy-balanced sampling dataset should be used?")
                        )

    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}_({}_bal)'.format(
        args.loss, args.num_epochs, args.batch_size, args.lr, args.model_type, args.sampling_strategy)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(args.save_dir, model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path,exist_ok=True)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = W2V2_AASIST_Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    loss_func=MultiTaskWeightedLoss(args.model_type,args.use_learnable_weight)
    loss_func=loss_func.to(device)

    #set Adam optimizer
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': loss_func.parameters()}
            ], lr=args.lr,weight_decay=args.weight_decay)
        
    if args.sampling_strategy=="VQ":
        train_txt="train_mult_learn_bal_vq.txt"
        valid_txt="dev_mult_learn_bal_vq.txt"
        test_txt="test_mult_learn_bal_vq.txt"
    elif args.sampling_strategy=="AUX":
        train_txt="train_mult_learn_bal_aux.txt"
        valid_txt="dev_mult_learn_bal_aux.txt"
        test_txt="test_mult_learn_bal_aux.txt"
    else:
        train_txt="train_mult_learn_bal_dec.txt"
        valid_txt="dev_mult_learn_bal_dec.txt"
        test_txt="test_mult_learn_bal_dec.txt"
    
    # define train dataloader
    d_label_trn,file_train = genSpoof_list(dir_meta =  os.path.join(args.metadata_path,train_txt),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set = Dataset_train(args, list_IDs=file_train,
        labels=d_label_trn,
        base_dir=args.base_dir,
        algo=args.algo
    )

    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.metadata_path,valid_txt),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_train(args, list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=args.base_dir,
        algo=args.algo
    )

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    # define evaluation dataloader
    file_eval = genSpoof_list(
        dir_meta=os.path.join(args.metadata_path,test_txt),
        is_train=False,
        is_eval=True
    )
    eval_set = Dataset_eval(
        list_IDs=file_eval,
        base_dir=args.base_dir
    )

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    # Initialize best values
    best_dev_loss = float('inf')
    best_dev_binary_eer = float('inf')
    best_dev_multi_acc = 0
    best_eval_binary_eer = float('inf')
    best_eval_multi_acc = 0  # Add for eval accuracy

    best_dev_loss_epoch = -1 
    best_dev_binary_epoch = -1
    best_dev_multi_epoch = -1
    best_eval_binary_epoch = -1
    best_eval_multi_epoch = -1  # Add for eval accuracy

    print("Training for {} epochs...".format(num_epochs))

    last_epoch = get_last_epoch(model_save_path)
    start_epoch = last_epoch + 1 if last_epoch >= 0 else 0

    # Load previous model if exists
    if start_epoch > 0:
        last_model_path = os.path.join(model_save_path, f'epoch_{last_epoch}.pth')
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        print(f'Resuming from epoch {start_epoch}')

    print(f"Training from epoch {start_epoch+1} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        running_loss = train_epoch(train_loader, model, args, optimizer, device,loss_func)
        
        # Dev evaluation
        val_loss, metrics = evaluate_metrics(dev_loader, model, device,loss_func)
        
        # Eval evaluation    
        produce_evaluation_file(eval_set, model, model_save_path, epoch)
        protocol_file = os.path.join(args.metadata_path, test_txt)
        
        eval_binary_eer = compute_eval_eer(os.path.join(  model_save_path, f"bin_score_epoch_{epoch}.txt"), protocol_file) if args.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"] else None
        eval_as_acc = compute_multi_accuracy(os.path.join(model_save_path, f"aux_score_epoch_{epoch}.txt"), protocol_file,type="AS") if args.model_type in ["S_AUX", "D_AUX", "M1", "M2"] else None
        eval_ds_acc = compute_multi_accuracy(os.path.join(model_save_path, f"dec_score_epoch_{epoch}.txt"), protocol_file,type="DS") if args.model_type in ["S_DEC", "D_DEC", "M1", "M2"] else None
        eval_qs_acc = compute_multi_accuracy(os.path.join(model_save_path, f" vq_score_epoch_{epoch}.txt"), protocol_file,type="QS") if args.model_type in ["S_VQ", "D_VQ", "M1", "M2"] else None

        val_acc_list = [acc for acc in [metrics[1], metrics[2], metrics[3]] if acc is not None]
        val_binary_eer = metrics[0]
        val_avg_acc = sum(val_acc_list) / len(val_acc_list) if val_acc_list else None
        
        eval_acc_list = [acc for acc in [eval_as_acc, eval_ds_acc, eval_qs_acc] if acc is not None]
        eval_avg_acc = sum(eval_acc_list) / len(eval_acc_list) if eval_acc_list else None
        
        # Update best values
        if val_loss < best_dev_loss:
            best_dev_loss = val_loss
            best_dev_loss_epoch = epoch + 1

        if args.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"]:
            if val_binary_eer < best_dev_binary_eer:
                best_dev_binary_eer = val_binary_eer
                best_dev_binary_epoch = epoch + 1
    
            if eval_binary_eer < best_eval_binary_eer:
                best_eval_binary_eer = eval_binary_eer
                best_eval_binary_epoch = epoch + 1

        if args.model_type not in ["S_BIN"]:
            if val_avg_acc > best_dev_multi_acc:
                best_dev_multi_acc = val_avg_acc
                best_dev_multi_epoch = epoch + 1
        
            if eval_avg_acc > best_eval_multi_acc: 
                best_eval_multi_acc = eval_avg_acc
                best_eval_multi_epoch = epoch + 1

        # Save model
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}.pth'))
        
        # Log to tensorboard
        writer.add_scalar('Train/Loss', running_loss, epoch)
        writer.add_scalar('Dev/Loss', val_loss, epoch)
        if args.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"]:
            writer.add_scalar('Dev/Binary_EER', val_binary_eer, epoch)
            writer.add_scalar('Eval/Binary_EER', eval_binary_eer, epoch)  
        if args.model_type not in ["S_BIN"]:
            writer.add_scalar('Dev/Average_Acc', val_avg_acc, epoch)
            writer.add_scalar('Eval/Average_Acc', eval_avg_acc, epoch)  
        
        # Print results per epoch
        msg = f"Epoch {epoch+1} - Train Loss: {running_loss:.4f} - Dev Loss: {val_loss:.4f} "
        
        if args.model_type == "S_BIN":
            msg += f"- Dev Binary EER: {val_binary_eer*100:.2f}% - Eval Binary EER: {eval_binary_eer*100:.2f}%"
        
        elif args.model_type in ["S_AUX", "S_DEC", "S_VQ"]:
            msg += (f"- Dev Accuracy: {val_avg_acc*100:.2f}% "
                    f"- Eval Accuracy: {eval_avg_acc*100:.2f}% ")
        
        elif args.model_type.startswith("D"):
            msg += (f"- Dev Binary EER: {val_binary_eer*100:.2f}% - Eval Binary EER: {eval_binary_eer*100:.2f}% "
                    f"- Dev Accuracy: {val_avg_acc*100:.2f}% - Eval Accuracy: {eval_avg_acc*100:.2f}% ")
        
        elif args.model_type == "M1":
            msg += (f"- Dev Binary EER: {val_binary_eer*100:.2f}% - Eval Binary EER: {eval_binary_eer*100:.2f}% "
                    f"- Dev avg. Accuracy: {val_avg_acc*100:.2f}% "
                    f"(AUX:{val_acc_list[0]*100:.2f}%, DEC:{val_acc_list[1]*100:.2f}%, VQ:{val_acc_list[2]*100:.2f}%) "
                    f"- Eval avg. Accuracy: {eval_avg_acc*100:.2f}% "
                    f"(AUX:{eval_as_acc*100:.2f}%, DEC:{eval_ds_acc*100:.2f}%, VQ:{eval_qs_acc*100:.2f}%)")
        
        elif args.model_type == "M2":
            msg += (f"- Dev avg. Accuracy: {val_avg_acc*100:.2f}% "
                    f"(AUX:{val_acc_list[0]*100:.2f}%, DEC:{val_acc_list[1]*100:.2f}%, VQ:{val_acc_list[2]*100:.2f}%) "
                    f"- Eval avg. Accuracy: {eval_avg_acc*100:.2f}% "
                    f"(AUX:{eval_as_acc*100:.2f}%, DEC:{eval_ds_acc*100:.2f}%, VQ:{eval_qs_acc*100:.2f}%)")
        
        print(msg)

    # Training completed
    print('\nTraining completed!')
    print(f'Best Dev Loss: {best_dev_loss:.4f} (Epoch {best_dev_loss_epoch})')

    if args.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"]:
        print(f'Best Dev Binary EER: {best_dev_binary_eer*100:.2f}% (Epoch {best_dev_binary_epoch})')
        print(f'Best Eval Binary EER: {best_eval_binary_eer*100:.2f}% (Epoch {best_eval_binary_epoch})')
    if args.model_type not in ["S_BIN"]:
        print(f'Best Dev Multi Acc: {best_dev_multi_acc*100:.2f}% (Epoch {best_dev_multi_epoch})')
        print(f'Best Eval Multi Acc: {best_eval_multi_acc*100:.2f}% (Epoch {best_eval_multi_epoch})')
