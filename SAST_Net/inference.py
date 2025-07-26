import argparse,os
import torch

from dataset import genSpoof_list, Dataset_eval
from model_SAST_NET import SAST_Net
from utils import compute_f1_score,compute_eval_eer,produce_evaluation_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to test.txt metadata')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory for eval data')
    parser.add_argument('--eval_output', type=str, default="Results",
                        help='Directory to save the evaluation result')
    parser.add_argument('--task', type=str, required=True,
                        help='Bin, AUX, DEC , VQ')
    parser.add_argument('--dataset_type', type=str, required=True,
                        help='CoRS or CoSG dataset')
    parser.add_argument('--use_SSL_feat', action='store_true', default=False)
    parser.add_argument('--use_semantic', action='store_true', default=False)
    parser.add_argument('--use_multi_decoder', action='store_true', default=True)
    parser.add_argument('--remove_silence', action='store_true', default=False)
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    torch.manual_seed(42)

    if args.use_multi_decoder:
        if args.task=="AUX" or args.task=="VQ":
            num_decoder=4
        elif args.task=="DEC":
            num_decoder=3
        else:
            num_decoder=2
    else:
        num_decoder=1

    model=SAST_Net(args,device,task=args.task,num_decoder=num_decoder,use_SSL_feat=args.use_SSL_feat, use_semantic=args.use_semantic).to(device)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print('Model loaded : {}'.format(args.model_path))

    is_COSG=args.dataset_type=="CoSG"

        
    # Prepare evaluation data
    file_eval = genSpoof_list(
        dir_meta=args.metadata_path,
        is_train=False,
        is_eval=True,
        is_SLM_ALL=is_COSG
    )
    eval_set = Dataset_eval(
        list_IDs=file_eval,
        base_dir=args.base_dir,
        return_raw_wav=True,
        remove_silence=args.remove_silence
    )

    # Produce evaluation scores and save to file
    produce_evaluation_file(eval_set, model, device, args.eval_output) 
    
    # Compute metric
    with open(os.path.join(args.eval_output,f"{args.task}_task_{args.dataset_type}_result.txt"), "w") as f:
        if args.task!="Bin":
            eval_f1 = compute_f1_score(os.path.join(args.eval_output,"tmp_scores.txt"), args.metadata_path,is_COSG,args.task)
            os.remove(os.path.join(args.eval_output,"tmp_scores.txt"))
            print(f'Eval F1: {eval_f1 * 100:.2f}%')
            f.write(f"SAST-Net on {args.task} source tracing task ({args.dataset_type} dataset) : {eval_f1 * 100:.2f}% F1 score.")
            
        else:
            eval_eer=compute_eval_eer(os.path.join(args.eval_output,"tmp_scores.txt"), args.metadata_path, is_COSG)
            os.remove(os.path.join(args.eval_output,"tmp_scores.txt"))
            print(f'Eval EER: {eval_eer * 100:.2f}%')
            f.write(f"SAST-Net on binary spoof detection task ({args.dataset_type} dataset) : {eval_eer * 100:.2f}% EER.")


        

