import os
import numpy as np
import torch,torchaudio
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random
from collections import defaultdict




def genSpoof_list(dir_meta, is_train=False, is_eval=False, is_SLM_ALL=False):
   d_meta_binary = {}
   d_meta_multi_as = {}
   d_meta_multi_ds = {}
   d_meta_multi_qs = {}
   file_list = []

   multi_label_map_as = {
       'REAL': 0,
       'Trad': 1, 
       'SEM': 2,
       'Disent': 3
   }
   multi_label_map_ds = {
       'REAL': 0,
       'TIME': 1, 
       'FREQ': 2
   }
   multi_label_map_qs = {
        'REAL': 0,
        'MVQ': 1,
        'SVQ': 2,
        'SQ': 3
   }

   with open(dir_meta, 'r') as f:
       l_meta = f.readlines()

   if is_train:
       for line in l_meta:
           parts = line.strip().split()
           if len(parts) >= 5:
               key = parts[1]
               multi_label_as = parts[-4]  # 倒数第四列是四分类标签as
               multi_label_ds = parts[-3]  # 倒数第三列是三分类标签ds
               multi_label_qs = parts[-2]  # 倒数第二列是四分类标签qs
               binary_label = parts[-1] # 最后一列是二分类标签
               file_list.append(key)
               d_meta_binary[key] = 0 if binary_label == 'bonafide' else 1
               d_meta_multi_as[key] = multi_label_map_as[multi_label_as]
               d_meta_multi_ds[key] = multi_label_map_ds[multi_label_ds]
               d_meta_multi_qs[key] = multi_label_map_qs[multi_label_qs]
       return (d_meta_binary, d_meta_multi_as,d_meta_multi_ds,d_meta_multi_qs), file_list

   elif is_eval:
       for line in l_meta:
           if is_SLM_ALL==False:
               key = line.strip().split()[1]
               file_list.append(key)
           else:
               codec = line.strip().split()[0]
               key = line.strip().split()[1]
               file_list.append(f"{codec}/{key}")
           
       return file_list

   else:
       for line in l_meta:
           parts = line.strip().split()
           if len(parts) >= 5:
               key = parts[1]
               multi_label_as = parts[-4]  # 倒数第四列是四分类标签as
               multi_label_ds = parts[-3]  # 倒数第三列是三分类标签ds
               multi_label_qs = parts[-2]  # 倒数第二列是四分类标签qs
               binary_label = parts[-1] # 最后一列是二分类标签
               file_list.append(key)
               d_meta_binary[key] = 0 if binary_label == 'bonafide' else 1
               d_meta_multi_as[key] = multi_label_map_as[multi_label_as]
               d_meta_multi_ds[key] = multi_label_map_ds[multi_label_ds]
               d_meta_multi_qs[key] = multi_label_map_qs[multi_label_qs]
       return (d_meta_binary, d_meta_multi_as,d_meta_multi_ds,d_meta_multi_qs), file_list

def pad(x, max_len=64600):
   x_len = x.shape[0]
   if x_len >= max_len:
       return x[:max_len]
   num_repeats = int(max_len / x_len) + 1
   padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
   return padded_x

def load_audio(file_path, sr=16000):
   X, fs = librosa.load(file_path, sr=None)
   if fs != sr:
       X = librosa.resample(X, orig_sr=fs, target_sr=sr)
   return X, sr

def remove_silence(audio: np.ndarray, sample_rate: int=16000, frame_duration_ms: int = 30, aggressiveness: int = 3) -> np.ndarray:
    """
    Removes silence from an audio waveform using WebRTC VAD.
    
    Args:
        audio (np.ndarray): Input audio waveform as a NumPy array.
        sample_rate (int): Sample rate of the audio.
        frame_duration_ms (int): Frame duration in milliseconds (default: 30 ms).
        aggressiveness (int): VAD aggressiveness level (0-3, default: 3, more aggressive means more silence removed).
    
    Returns:
        np.ndarray: Audio waveform with silence removed.
    """
    vad = webrtcvad.Vad(aggressiveness)
    
    # Convert float audio [-1,1] to int16 [-32768, 32767]
    audio_int16 = (audio * 32767).astype(np.int16)
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # Number of samples per frame
    num_frames = len(audio_int16) // frame_size
    
    voiced_frames = []
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio_int16[start:end].tobytes()
        
        if vad.is_speech(frame, sample_rate):
            voiced_frames.append(audio_int16[start:end])
    
    if not voiced_frames:
        return np.array([], dtype=np.float32)  # Return an empty array if no voiced frames found
    
    voiced_audio = np.concatenate(voiced_frames, axis=0)
    return voiced_audio.astype(np.float32) / 32767  # Convert back to float32

def norm_fbank(fbank, remove_silence=False):
    norm_mean= -8.7086 if remove_silence==False else -6.1276
    norm_std= 4.4163 if remove_silence==False else 3.8073
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

class Dataset_train(Dataset):
   def __init__(self, args, list_IDs, labels, base_dir, algo,return_raw_wav=False,is_valid=False,remove_silence=False, audio_len=82200):
       self.list_IDs = list_IDs

       self.labels_binary = labels[0]
       self.labels_multi_as = labels[1]
       self.labels_multi_ds = labels[2]
       self.labels_multi_qs = labels[3]
       
       self.base_dir = base_dir
       self.algo = algo
       self.args = args
       self.return_raw_wav=return_raw_wav
       self.cut = audio_len   
       self.remove_silence=remove_silence
       if remove_silence:
           import webrtcvad
    
   def __len__(self):
       return len(self.list_IDs) 

   def read_audio(self,uttid):
       possible_extensions = ['.flac', '.wav']
       for ext in possible_extensions:
           file_path = os.path.join(self.base_dir, uttid + ext)
           if os.path.isfile(file_path):
               file_found = True
               break
    
       if not file_found:
           raise FileNotFoundError(f"File {uttid} not found in {self.base_dir}")
    
       #X, sr = torchaudio.load(file_path)
       X2, sr = load_audio(file_path, sr=16000)
       if self.remove_silence:
           X=remove_silence(X2)
           if len(X)==0:
               X=X2
           else:
               X2=X
       if self.algo:
           X2 = process_Rawboost_feature(X2, sr, self.args, self.algo)
       X2_pad = pad(X2, self.cut)
       x2_inp = Tensor(X2_pad)
       return x2_inp
       
   def __getitem__(self, index):
       ## return melspec:[Batch,Channel=1,Time_Bins=512,Freq_Bins=128] , binary_label, aux_label, decoder_label, vq_label
       utt_id = self.list_IDs[index] 
       x_inp = self.read_audio(utt_id)
       binary_target = self.labels_binary[utt_id]
       multi_target_as = self.labels_multi_as[utt_id]
       multi_target_ds = self.labels_multi_ds[utt_id]
       multi_target_qs = self.labels_multi_qs[utt_id]

       
       if self.return_raw_wav:
           return x_inp , binary_target, multi_target_as,multi_target_ds,multi_target_qs

           
       ## ,Shape of x_inp=(82200),  Shape of fbank=(512,128)
       fbank = torchaudio.compliance.kaldi.fbank(x_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
       fbank=norm_fbank(fbank,remove_silence=self.remove_silence)
       return fbank.unsqueeze(0), binary_target, multi_target_as,multi_target_ds,multi_target_qs

           
class Dataset_eval(Dataset):
   def __init__(self, list_IDs, base_dir,return_raw_wav=False,remove_silence=False,audio_len=82200):
       self.list_IDs = list_IDs
       self.base_dir = base_dir
       self.cut = audio_len
       self.return_raw_wav=return_raw_wav
       self.remove_silence=remove_silence
       if remove_silence:
           import webrtcvad

   def __len__(self):
       return len(self.list_IDs)

   def __getitem__(self, index):
       utt_id = self.list_IDs[index]
       possible_extensions = ['.flac', '.wav']
       file_found = False
       
       for ext in possible_extensions:
           file_path = os.path.join(self.base_dir, utt_id + ext)
           if os.path.isfile(file_path):
               file_found = True
               break
               
       if not file_found:
           raise FileNotFoundError(f"File {utt_id} not found")
           
       #X, sr = torchaudio.load(file_path)
       X, sr = load_audio(file_path, sr=16000)
       if sr != 16000:
           X = librosa.resample(X, orig_sr=sr, target_sr=16000)
    
       if self.remove_silence:
           X=remove_silence(X)
            
            
       X_pad = pad(X, self.cut)
       x_inp = torch.FloatTensor(X_pad)
       if self.return_raw_wav:
           return x_inp,utt_id
           
       fbank = torchaudio.compliance.kaldi.fbank(x_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
       fbank=norm_fbank(fbank,remove_silence=self.remove_silence)
       
       return fbank.unsqueeze(0), utt_id


#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
