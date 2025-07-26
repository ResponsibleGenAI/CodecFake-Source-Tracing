import torch
import os,sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

#REAL:0 Trad:1 ; SEM: 2; DISENT: 3
AS_CORS_dict={
    'SpeechTokenizer_hubert_avg':2,
    'DAC24':1,
    'HiFi_Codec_16k_320d_large_universal':1,
    'Encodec_6b24k':1,
    'Encodec_24b24k':1,
    'FunCodec_zh_en-general-16k-nq32ds320':1,
    'FACodec_encodec-decoder-v2_16k':3,
    'llm_codec':2,
    'snac_24khz':1,
    'snac_44khz':1,
    'xcodec_hubert_librispeech':2,
    'xcodec_hubert_libirispeech':2,
    'xcodec_hubert_general_audio':2,
    'xocdec_hubert_general_audio':2,
    'bigcodec':1,
    'ticodec_1g1r':3,
    'ticodec_1g2r':3,
    'ticodec_1g4r':3,
    'mimicodec':2,
    'mini':2,
    'sqcodec50dim9':1,
    'SemantiCodec_1.40kbps_16k':2,
    'SemantiCodec_0.70kbps_16k':2,
    'SemantiCodec_0.35kbps_16k':2,
    'FunCodec_en-libritts-16k-gr1nq32ds320':1,
    'wavtokenizer_small_320_24k_4096':1,
    'wavtokenizer_medium_320_24k_4096':1,
    'languagecodec':1,
    'vocos_encodec_6':1,
    'vocos_encodec_12':1,
    'socodec_16384x4_120ms_16khz_chinese':3,
    'audiodec_24k_320d':1,
    'spectralcodecs':1,
    "hilcodec_speech":1,
    'ALL':-1
}

#REAL:0 TIME:1 ; FREQ: 2; 
DS_CORS_dict={
    'SpeechTokenizer_hubert_avg':1,
    'DAC24':1,
    'HiFi_Codec_16k_320d_large_universal':1,
    'Encodec_6b24k':1,
    'Encodec_24b24k':1,
    'FunCodec_zh_en-general-16k-nq32ds320':1,
    'FACodec_encodec-decoder-v2_16k':1,
    'llm_codec':1,
    'snac_24khz':1,
    'snac_44khz':1,
    'xcodec_hubert_librispeech':1,
    'xcodec_hubert_libirispeech':1,
    'xcodec_hubert_general_audio':1,
    'xocdec_hubert_general_audio':1,
    'bigcodec':1,
    'ticodec_1g1r':1,
    'ticodec_1g2r':1,
    'ticodec_1g4r':1,
    'mimicodec':1,
    'mini':1,
    'sqcodec50dim9':1,
    'SemantiCodec_1.40kbps_16k':2,
    'SemantiCodec_0.70kbps_16k':2,
    'SemantiCodec_0.35kbps_16k':2,
    'FunCodec_en-libritts-16k-gr1nq32ds320':2,
    'wavtokenizer_small_320_24k_4096':2,
    'wavtokenizer_medium_320_24k_4096':2,
    'languagecodec':2,
    'vocos_encodec_6':2,
    'vocos_encodec_12':2,
    'socodec_16384x4_120ms_16khz_chinese':2,
    'audiodec_24k_320d':2,
    'spectralcodecs':2,
    "hilcodec_speech":1,
    'ALL':-1
}

#REAL:0 MVQ:1 ; SVQ: 2; SQ:3
QS_CORS_dict={
    'SpeechTokenizer_hubert_avg':1,
    'DAC24':1,
    'HiFi_Codec_16k_320d_large_universal':1,
    'Encodec_6b24k':1,
    'Encodec_24b24k':1,
    'FunCodec_zh_en-general-16k-nq32ds320':1,
    'FACodec_encodec-decoder-v2_16k':1,
    'llm_codec':1,
    'snac_24khz':1,
    'snac_44khz':1,
    'xcodec_hubert_librispeech':1,
    'xcodec_hubert_libirispeech':1,
    'xcodec_hubert_general_audio':1,
    'xocdec_hubert_general_audio':1,
    'bigcodec':2,
    'ticodec_1g1r':2,
    'ticodec_1g2r':1,
    'ticodec_1g4r':1,
    'mimicodec':1,
    'mini':1,
    'sqcodec50dim9':3,
    'SemantiCodec_1.40kbps_16k':1,
    'SemantiCodec_0.70kbps_16k':1,
    'SemantiCodec_0.35kbps_16k':1,
    'FunCodec_en-libritts-16k-gr1nq32ds320':1,
    'wavtokenizer_small_320_24k_4096':2,
    'wavtokenizer_medium_320_24k_4096':2,
    'languagecodec':1,
    'vocos_encodec_6':1,
    'vocos_encodec_12':1,
    'socodec_16384x4_120ms_16khz_chinese':1,
    'audiodec_24k_320d':1,
    'spectralcodecs':3,
    "hilcodec_speech":1,
    'ALL':-1
}


#REAL:0 Trad:1 ; SEM: 2; DISENT: 3
AS_dict={
    "ELLAV" :1,
    "VALLE" :1,
    "GPST" :1,
    "UNIAUDIO" :1,
    "TACOLM" :1,
    "SPEECHX" :1,
    "RALLE" :1,
    "CLAMTTS" :1,
    "VIOLA" :1,
    "NS2" :1,
    "NS3" :3,
    "USLM" :2,
    "SIMPLESPEECH1" :1,
    "SIMPLESPEECH2" :1,
    "TI1G1R" :3,
    "SINGLECODEC" :3,
    "MASKGCT":1,
    "ALL":-1
}

#REAL:0 TIME:1 ; FREQ: 2; 
DS_dict={
    "ELLAV" :1,
    "VALLE" :1,
    "GPST" :1,
    "UNIAUDIO" :1,
    "TACOLM" :1,
    "SPEECHX" :1,
    "RALLE" :1,
    "CLAMTTS" :2,
    "VIOLA" :1,
    "NS2" :1,
    "NS3" :1,
    "USLM" :1,
    "SIMPLESPEECH1" :1,
    "SIMPLESPEECH2" :1,
    "TI1G1R" :1,
    "SINGLECODEC" :2,
    "MASKGCT":2,
    "ALL":-1
}

#REAL:0 MVQ:1 ; SVQ: 2; SQ:3
QS_dict={
    "ELLAV" :1,
    "VALLE" :1,
    "GPST" :1,
    "UNIAUDIO" :1,
    "TACOLM" :1,
    "SPEECHX" :1,
    "RALLE" :1,
    "CLAMTTS" :1,
    "VIOLA" :1,
    "NS2" :1,
    "NS3" :1,
    "USLM" :1,
    "SIMPLESPEECH1" :3,
    "SIMPLESPEECH2" :3,
    "TI1G1R" :2,
    "SINGLECODEC" :2,
    "MASKGCT":1,
    "ALL":-1
}

class_AS={0:"Real",1:"None",2:"SEM",3:"Disentangle"}
class_DS={0:"Real",1:"Time",2:"Frequency"}
class_QS={0:"Real",1:"MVQ",2:"SVQ",3:"SQ"}


def save_cm(y_true, y_pred, save_path, class_label=None):
    """
    Save confusion matrix as a heatmap.

    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.
        save_path (str): Path to save the confusion matrix image.
        class_label (dict or list, optional): Mapping of class indices to labels. 
                                              If None, use numeric labels.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Handle class labels for display
    if class_label is None:
        class_label = [str(i) for i in range(len(set(y_true) | set(y_pred)))]
        all_classes = np.unique(y_true + y_pred)
    elif isinstance(class_label, dict):
        all_classes = list(class_label.keys())
        class_label = [class_label[i] for i in all_classes]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=all_classes)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Confusion Matrix (Heatmap)")
    tick_marks = np.arange(len(class_label))
    plt.xticks(tick_marks, class_label, rotation=45)
    plt.yticks(tick_marks, class_label)

    # Annotate heatmap with values
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save heatmap
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def load_scores_and_labels(score_file, protocol_file, is_COSG=False):
    scores = {}
    labels = {}

    with open(score_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            utt_id=tokens[0]
            s=tokens[1] 
            scores[utt_id] = float(s)
            
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[1]
            codec=parts[0] if is_COSG else utt_id[9:]
            label = parts[-1]
            if is_COSG:
                labels[f"{codec}/{utt_id}"] = 0 if label == 'bonafide' else 1
            else:
                labels[utt_id] = 0 if label == 'bonafide' else 1


    sorted_utt_ids = sorted(scores.keys())
    y_scores = []
    y_true = []

    for utt_id in sorted_utt_ids:
        if utt_id in labels:
            y_scores.append(scores[utt_id])
            y_true.append(labels[utt_id])

    print(f"Processed {len(y_scores)} trials")
    return np.array(y_true), np.array(y_scores)
    

def compute_det_curve(target_scores, nontarget_scores):
    
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_eval_eer(score_file, protocol_file, is_COSG=False):
    y_true, y_scores = load_scores_and_labels(score_file, protocol_file,is_COSG)

    if len(y_true) != len(y_scores):
        print("Warning: Mismatch between number of scores and labels!")
        return None

    target_scores = y_scores[y_true == 0]
    nontarget_scores = y_scores[y_true == 1]

    print(f"Number of bonafide trials: {len(target_scores)}")
    print(f"Number of spoof trials: {len(nontarget_scores)}")

    eer, threshold = compute_eer(target_scores, nontarget_scores)
    print(threshold)
    return eer

def compute_f1_score(score_file, protocol_file,is_COSG=True, task=""):
    scores = {}
    labels = {}

    with open(score_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            utt_id=tokens[0]
            s=tokens[1:]
            scores[utt_id] = s
            
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[1]
            codec=parts[0] if is_COSG else utt_id[9:]
            if task=="AUX":
                map_dict=AS_dict if is_COSG else AS_CORS_dict
            elif task=="DEC":
                map_dict=DS_dict if is_COSG else DS_CORS_dict
            elif task=="VQ":
                map_dict=QS_dict if is_COSG else QS_CORS_dict
            label = parts[-1]
            if is_COSG:
                labels[f"{codec}/{utt_id}"] = 0 if label == 'bonafide' else map_dict[codec]
            else:
                labels[utt_id] = 0 if label == 'bonafide' else map_dict[codec]

    sorted_utt_ids = sorted(scores.keys())
    y_scores = []
    y_true = []

    for utt_id in sorted_utt_ids:
        if utt_id in labels:
            sc=[]
            for x in scores[utt_id]:
                sc.append(float(x))
            y_scores.append(np.array(sc))
            y_true.append(labels[utt_id])

    y_pred = np.argmax(y_scores, axis=1)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Processed {len(y_scores)} trials")
    print(f"F1 Score: {f1:.4f}")
    
    return f1

def produce_evaluation_file(dataset, model, device, save_dir):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()

    progress_bar = tqdm(data_loader, desc='Evaluating')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path=os.path.join(save_dir,"tmp_scores.txt")
    
    with open(save_path, 'w') as fh:
        for batch_x, utt_id in progress_bar:
            # sys.exit(0)
            batch_x = batch_x.to(device)

            with torch.no_grad():
                multi_out = model.predict(batch_x)
                multi_score = multi_out.softmax(dim=1).cpu().numpy()
                multi_score = multi_out.cpu().numpy()

                for f, cm in zip(utt_id, multi_score):
                    fh.write('{} {}\n'.format(f, " ".join(map(str, cm))))

    print('Scores saved to {}'.format(save_path))