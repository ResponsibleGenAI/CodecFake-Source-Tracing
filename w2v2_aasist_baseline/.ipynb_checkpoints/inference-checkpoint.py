import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import f1_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

from dataset import genSpoof_list,Dataset_train,Dataset_eval
from utils import compute_f1_score, compute_eval_eer
from model import W2V2_AASIST_Model

def save_result():
    with open(save_path, 'w') as fh:
        for f, cm in zip(utt_id, batch_score):
            fh.write('{} {}\n'.format(f, " ".join(map(str, cm))))

    print('Scores saved to {}'.format(save_path))


def produce_evaluation_file(dataset, model, args):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    device=model.device
    model_type = args.model_type
    
    save_dir=args.eval_output
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    trash_file=os.path.join(save_dir, "trash.txt")

    bin_sv_path=os.path.join(save_dir, "bin_score.txt") if model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"] else trash_file
    aux_sv_path=os.path.join(save_dir, "aux_score.txt") if model_type in ["S_AUX", "D_AUX", "M1", "M2"] else trash_file
    dec_sv_path=os.path.join(save_dir, "dec_score.txt") if model_type in ["S_DEC", "D_DEC", "M1", "M2"] else trash_file
    vq_sv_path=os.path.join(save_dir, "vq_score.txt") if model_type in ["S_VQ", "D_VQ", "M1", "M2"] else trash_file
    
    progress_bar = tqdm(data_loader, desc='Evaluating')
    with open(bin_sv_path, 'w') as f_bin, open(aux_sv_path, 'w') as f_aux, open(dec_sv_path, 'w') as f_dec, open(vq_sv_path, 'w') as f_vq:
        for batch_x, utt_id in progress_bar:
            batch_x = batch_x.to(device)
    
            with torch.no_grad():
                outputs = model(batch_x)

                if model_type!="M2":
                    binary_scores = softmax(outputs['binary'], dim=1)[:, 1].cpu().numpy() if model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1"] else None
                else:
                    a = softmax(outputs['as'], dim=1)[:, 1:].sum(dim=1)  # The spoof probability in the AUX task
                    d = softmax(outputs['ds'], dim=1)[:, 1:].sum(dim=1)  # The spoof probability in the DEC task
                    q = softmax(outputs['qs'], dim=1)[:, 1:].sum(dim=1)  # The spoof probability in the VQ task
                    binary_scores = 1 - torch.pow(a * d * q, 1/3).cpu().numpy() # Bonafide probability obtained from the three tasks 
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory for eval data')
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to meta data txt file ')
    parser.add_argument('--dataset_type', type=str, required=True, choices=["CoRS","CoSG"], 
                        help='Evaluation dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--w2v2_pretrain_path', type=str, default="./xlsr2_300m.pt",
                        help='Path to xlsr pretrained weight')
    parser.add_argument('--eval_output', type=str, default="Result",
                        help='Directory to save the evaluation result')
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

    args = parser.parse_args()
    os.makedirs(args.eval_output,exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # Load the model
    model = W2V2_AASIST_Model(args, device).to(device)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print('Model loaded : {}'.format(args.model_path))
    
    # Prepare evaluation data
    file_eval = genSpoof_list(
        dir_meta=args.metadata_path,
        is_train=False,
        is_eval=True,
        is_CoSG=args.dataset_type=="CoSG"
    )
    eval_set = Dataset_eval(list_IDs=file_eval, base_dir=args.base_dir)

    # Produce evaluation scores and save to file
    produce_evaluation_file(eval_set, model, args)
    
    # Compute Metrics and save to results.txt
    with open(os.path.join(args.eval_output, "results.txt"),"w") as f:
        if args.model_type in ["S_BIN", "D_VQ", "D_AUX", "D_DEC", "M1", "M2"]:
            eval_eer = compute_eval_eer(os.path.join(args.eval_output, "bin_score.txt"), args.metadata_path, is_CoSG=args.dataset_type=="CoSG")
            f.write(f"Equal Error Rate of {args.model_type} on {args.dataset_type} dataset: {eval_eer * 100:.2f}%\n")
        if args.model_type in ["S_AUX", "D_AUX", "M1", "M2"]:
            eval_f1 = compute_f1_score(os.path.join(args.eval_output, "aux_score.txt"), args.metadata_path, is_CoSG=args.dataset_type=="CoSG", task="AS")
            f.write(f"Auxiliary objective classification F1 score of {args.model_type} on the {args.dataset_type} dataset: {eval_f1 * 100:.2f}%\n")
        if args.model_type in ["S_DEC", "D_DEC", "M1", "M2"]:
            eval_f1 = compute_f1_score(os.path.join(args.eval_output, "dec_score.txt"), args.metadata_path, is_CoSG=args.dataset_type=="CoSG", task="DS")
            f.write(f"Decoder type classification F1 score of {args.model_type} on the {args.dataset_type} dataset: {eval_f1 * 100:.2f}%\n")
        if args.model_type in ["S_VQ", "D_VQ", "M1", "M2"]:
            eval_f1 = compute_f1_score(os.path.join(args.eval_output, "vq_score.txt"), args.metadata_path, is_CoSG=args.dataset_type=="CoSG", task="QS")
            f.write(f"Vector quantization classification F1 score of {args.model_type} on the {args.dataset_type} dataset: {eval_f1 * 100:.2f}%\n")
        

