import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml

from dataset import genSpoof_list, Dataset_train, Dataset_eval
from model_SAST_NET import SAST_Net,load_XLSR
from AutoWeightLoss import AutomaticWeightedLoss

from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm

from utils import load_scores_and_labels, compute_eer, compute_eval_eer

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def get_last_epoch(model_save_path):
    """获取最后一个epoch的编号"""
    epoch_files = [f for f in os.listdir(model_save_path) if f.startswith('epoch_') and f.endswith('.pth')]
    if not epoch_files:
        return -1
    epochs = [int(f.split('_')[1].split('.')[0]) for f in epoch_files]
    return max(epochs)

def evaluate_accuracy(dev_loader, model, device,awl,task,mask_ratio=0.3, use_SSL_feat=False):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    awl.eval()

    if task=="AUX" or task=="VQ":
        multi_weight = torch.FloatTensor([0.1, 0.3, 0.3, 0.3]).to(device)
    elif task=="DEC":
        multi_weight = torch.FloatTensor([0.2, 0.4, 0.4]).to(device)
    elif task=="Bin":
        multi_weight = torch.FloatTensor([0.5,0.5]).to(device)
    multi_criterion = nn.CrossEntropyLoss(weight=multi_weight)
    
    all_multi_scores = []
    all_multi_labels = []
    
    progress_bar = tqdm(dev_loader, desc='Validating')
    for batch_x, binary_target, multi_target_as,multi_target_ds,multi_target_qs in progress_bar:
        batch_size = batch_x.size(0)
        num_total += batch_size
            
        batch_x = batch_x.to(device)
        binary_target=binary_target.view(-1).type(torch.int64).to(device)
        multi_target_as = multi_target_as.view(-1).type(torch.int64).to(device)
        multi_target_ds = multi_target_ds.view(-1).type(torch.int64).to(device)
        multi_target_qs = multi_target_qs.view(-1).type(torch.int64).to(device)
            
        if task=="AUX":
            Label=multi_target_as
        if task=="DEC":
            Label=multi_target_ds
        if task=="VQ":
            Label=multi_target_qs
        if task=="Bin":
            Label=binary_target
        
        with torch.no_grad():
            out,reconstruction_loss = model(batch_x,Label,mask_ratio=mask_ratio)
            CE_loss = multi_criterion(out, Label)
            
            batch_loss = awl(reconstruction_loss,CE_loss )  
            val_loss += (batch_loss.item() * batch_size)
            
            multi_scores = out.softmax(dim=1).cpu().numpy() if task!="Bin" else out.cpu().numpy()
            
            all_multi_scores.extend(multi_scores)
            all_multi_labels.extend(Label.cpu().numpy())
        
        progress_bar.set_postfix({'Loss': f'{val_loss/num_total:.4f}'})
    
    val_loss /= num_total

    if task!="Bin":
        # compute accuracy
        multi_pred = np.argmax(all_multi_scores, axis=1)
        multi_acc = np.mean(multi_pred == all_multi_labels)
        return val_loss, multi_acc
    else:
        ts=np.vstack(all_multi_scores)[:,1][np.array(all_multi_labels)==0]
        nts=np.vstack(all_multi_scores)[:,1][np.array(all_multi_labels)==1]
        # compute EER
        eer, threshold= compute_eer(target_scores=ts, nontarget_scores=nts)
        return val_loss, eer

def compute_multi_accuracy(score_file, protocol_file,task):
    scores = []
    utt_ids = []
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt_ids.append(parts[0])
            scores.append(list(map(float, parts[1:])))  # [REAL, MVQ, SVQ, SQ]
            
    labels = {}
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                utt_id = parts[1]
                if task=="VQ":
                    multi_label = parts[-2]  # 倒数第二列
                    labels[utt_id] = {"REAL": 0, "MVQ": 1, "SVQ": 2, "SQ": 3}[multi_label]
                elif task=="DEC":
                    multi_label = parts[-3]  # 倒数第三列
                    labels[utt_id] = {"REAL": 0, "TIME": 1, "FREQ": 2}[multi_label]
                elif task=="AUX":
                    multi_label = parts[-4]  # 倒数第四列
                    labels[utt_id] = {"REAL": 0, "Trad": 1, "SEM": 2, "Disent": 3}[multi_label]
    
    predictions = []
    true_labels = []
    for utt_id, score in zip(utt_ids, scores):
        if utt_id in labels:
            predictions.append(np.argmax(score))
            true_labels.append(labels[utt_id])
    
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return accuracy


def produce_evaluation_file(dataset, model, device,multi_save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    
    with open(multi_save_path, 'w') as f_multi:
        for batch_x, utt_id in tqdm(data_loader, desc='Evaluating'):
            batch_x = batch_x.to(device)
            
            with torch.no_grad():
                # print(batch_x)
                multi_out = model.predict(batch_x)
                multi_scores = multi_out.softmax(dim=1).cpu().numpy()
                
                for f, m_scores in zip(utt_id, multi_scores):
                    # f_binary.write(f'{f} {b_score}\n')
                    f_multi.write(f'{f} {" ".join(map(str, m_scores))}\n')
    
    # print('Binary scores saved to {}'.format(binary_save_path))
    print('Multi scores saved to {}'.format(multi_save_path))

def train_epoch(train_loader, model, lr, optim,scheduler, device,awl,task,mask_ratio=0.3, use_SSL_feat=False):
    running_loss = 0
    num_total = 0.0
    model.train()
    awl.train()
    
    if task=="AUX" or task=="VQ":
        multi_weight = torch.FloatTensor([0.1, 0.3, 0.3, 0.3]).to(device)
    elif task=="DEC":
        multi_weight = torch.FloatTensor([0.2, 0.4, 0.4]).to(device)
    elif task=="Bin":
        multi_weight = torch.FloatTensor([0.5,0.5]).to(device)
    multi_criterion = nn.CrossEntropyLoss(weight=multi_weight)

    i=0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_x, binary_target, multi_target_as,multi_target_ds,multi_target_qs in progress_bar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        binary_target   =binary_target.view(-1).type(torch.int64).to(device)
        multi_target_as = multi_target_as.view(-1).type(torch.int64).to(device)
        multi_target_ds = multi_target_ds.view(-1).type(torch.int64).to(device)
        multi_target_qs = multi_target_qs.view(-1).type(torch.int64).to(device)

        if task=="AUX":
            Label=multi_target_as
        if task=="DEC":
            Label=multi_target_ds
        if task=="VQ":
            Label=multi_target_qs
        if task=="Bin":
            Label=binary_target

        out,reconstruction_loss = model(batch_x,Label,mask_ratio=mask_ratio)
            
        CE_loss = multi_criterion(out, Label)
        batch_loss = awl(reconstruction_loss,CE_loss )     
        running_loss += (batch_loss.item() * batch_size)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        weight=awl.params.detach().cpu().numpy()

        progress_bar.set_postfix({'AutoWeight Loss': f'{running_loss/num_total:.4f}',
                                          'Weight': ', '.join([f'{w:.4f}' for w in weight]),
                                          'Reconstruct Loss': f'{reconstruction_loss:.4f}',
                                          'CE Loss': f'{CE_loss:.4f}',
                                         })
        i+=1
        if i>10:break
       
    running_loss /= num_total
    scheduler.step()
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for CoRS dataset')
    parser.add_argument('--metadata_path', type=str, default='./metadata/training/', help='Change with path to user\'s LA database protocols directory address')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--mask_2D', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='AWL')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--use_SSL_feat', action='store_true', default=False)
    parser.add_argument('--use_semantic', action='store_true', default=False)
    parser.add_argument('--use_multi_decoder', action='store_true', default=False)
    
    
    
    # Auxiliary arguments
    parser.add_argument('--save_dir', type=str,
                        default="models_SAST_Net", help='Model checkpoint save directory')
    parser.add_argument('--task', type=str, default='AUX',choices=['Bin','AUX', 'DEC','VQ'], help='Classification task')
    parser.add_argument("--sampling_strategy", default="AUX",
                            choices=["VQ", "AUX", "DEC"],
                            help=("Which taxonomy-balanced sampling dataset should be used?")
                        )
    parser.add_argument('--load_tuned_weight', action='store_true', default=False)
    parser.add_argument('--tuned_weight_path', default="./Pretrain_weight/tuned_weight.pth", type=str, help='Path to trained wav2vec2_AASIST checkpoint')
    
    
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
        args.loss, args.num_epochs, args.batch_size, args.lr, args.task, args.sampling_strategy)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(args.save_dir, model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path,exist_ok=True)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    if args.use_multi_decoder:
        if args.task=="AUX" or args.task=="VQ":
            num_decoder=4
        elif args.task=="DEC":
            num_decoder=3
        else:
            num_decoder=2
    else:
        num_decoder=1
    
    model = SAST_Net(args,device,task=args.task,num_decoder=num_decoder,use_SSL_feat=args.use_SSL_feat, use_semantic=args.use_semantic, mask_2D=args.mask_2D)
    if args.use_SSL_feat and args.load_tuned_weight:
        load_XLSR(args.tuned_weight_path,model)
    
    awl=AutomaticWeightedLoss(2)

    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    awl=awl.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.AdamW([
                {'params': model.parameters(),"lr": args.lr, "weight_decay": args.weight_decay},
                {'params': awl.parameters(), 'weight_decay': 0} 
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs) 

    
    if args.sampling_strategy=="VQ" :
        train_txt="train_mult_learn_bal_vq.txt"
        dev_txt="dev_mult_learn_bal_vq.txt"
        test_txt="test_mult_learn_bal_vq.txt"
    elif args.sampling_strategy=="AUX" :
        train_txt="train_mult_learn_bal_dec.txt"
        dev_txt="dev_mult_learn_bal_dec.txt"
        test_txt="test_mult_learn_bal_dec.txt"
    elif args.sampling_strategy=="AS" :
        train_txt="train_mult_learn.txt"
        dev_txt="dev_mult_learn.txt"
        test_txt="test_mult_learn.txt"


    # define train dataloader
    d_label_trn,file_train = genSpoof_list(dir_meta =  os.path.join(args.metadata_path,train_txt),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set = Dataset_train(args, list_IDs=file_train,
        labels=d_label_trn,
        base_dir=args.base_dir,
        algo=args.algo,
        return_raw_wav=(args.use_SSL_feat or args.use_semantic),
        is_valid=False,
    )

    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)

    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.metadata_path,dev_txt),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_train(args, list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=args.base_dir,
        algo=args.algo,
        return_raw_wav=(args.use_SSL_feat or args.use_semantic),
        is_valid=True,
    )

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    # evaluation set
    file_eval = genSpoof_list(
        dir_meta=os.path.join(args.metadata_path,test_txt),
        is_train=False,
        is_eval=True
    )
    eval_set = Dataset_eval(
        list_IDs=file_eval,
        base_dir=args.base_dir,
        return_raw_wav=(args.use_SSL_feat or args.use_semantic)
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
        
        running_loss = train_epoch(train_loader, model, args.lr, optimizer, scheduler, device,awl,args.task,args.mask_ratio, use_SSL_feat=args.use_SSL_feat)
        
        # Dev evaluation
        val_loss, val_metric = evaluate_accuracy(dev_loader, model, device,awl,args.task,args.mask_ratio,  use_SSL_feat=args.use_SSL_feat)

        # Eval evaluation  
        eval_multi_file = os.path.join(model_save_path, f'eval_scores_epoch_{epoch}.txt')
        produce_evaluation_file(eval_set, model, device, eval_multi_file)
        protocol_file = os.path.join(args.metadata_path,test_txt)

        # Compute eval multi-class accuracy
        if args.task!="Bin":
            eval_multi_acc = compute_multi_accuracy(eval_multi_file, protocol_file,args.task)
        else:
            eval_eer = compute_eval_eer(eval_multi_file,protocol_file)
        
        # Update best values
        if val_loss < best_dev_loss:
            best_dev_loss = val_loss
            best_dev_loss_epoch = epoch + 1
            
        if args.task!="Bin":    
            if val_metric > best_dev_multi_acc:
                best_dev_multi_acc = val_metric
                best_dev_multi_epoch = epoch + 1
    
            if eval_multi_acc > best_eval_multi_acc:  # Update best eval accuracy
                best_eval_multi_acc = eval_multi_acc
                best_eval_multi_epoch = epoch + 1
        else:
            if val_metric< best_dev_binary_eer:
                best_dev_binary_eer = val_metric
                best_dev_binary_epoch = epoch + 1
    
            if eval_eer < best_eval_binary_eer:  # Update best eval accuracy
                best_eval_binary_eer = eval_eer
                best_eval_binary_epoch = epoch + 1

        # Save model
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}.pth'))
        
        # Print results
        if args.task!="Bin": 
            print(f'Epoch {epoch+1} - Train Loss: {running_loss:.4f} '
                f'- Dev Loss: {val_loss:.4f} '
                f'- Dev Multi Acc: {val_metric*100:.2f}% '
                f'- Eval Multi Acc: {eval_multi_acc*100:.2f}%')
        else:
            print(f'Epoch {epoch+1} - Train Loss: {running_loss:.4f} '
                f'- Dev Loss: {val_loss:.4f} '
                f'- Dev EER: {val_metric*100:.2f}% '
                f'- Eval EER: {eval_eer*100:.2f}%')
            
    # Training completed
    print('\nTraining completed!')
    print(f'Best Dev Loss: {best_dev_loss:.4f} (Epoch {best_dev_loss_epoch})')
    print(f'Best Dev Multi Acc: {best_dev_multi_acc*100:.2f}% (Epoch {best_dev_multi_epoch})') if args.task!="Bin" else print(f'Best Dev Binary EER: {best_dev_binary_eer*100:.2f}% (Epoch {best_dev_binary_epoch})')
    print(f'Best Eval Multi Acc: {best_eval_multi_acc*100:.2f}% (Epoch {best_eval_multi_epoch})') if args.task!="Bin" else print(f'Best Eval Binary EER: {best_eval_binary_eer*100:.2f}% (Epoch {best_eval_binary_epoch})')
