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
import webrtcvad

def gen_list(dir_meta,is_SLM_ALL=False):
   file_list=[]
   with open(dir_meta, 'r') as f:
       l_meta = f.readlines()
   for line in l_meta:
       if is_SLM_ALL==False:
           key = line.strip().split()[1]
           file_list.append(key)
       else:
           codec = line.strip().split()[0]
           key = line.strip().split()[1]
           label= line.strip().split()[-1]
           file_list.append(f"{codec}/{key}/{label}")
       
   return file_list



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
               d_meta_binary[key] = 1 if binary_label == 'bonafide' else 0
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
               d_meta_binary[key] = 1 if binary_label == 'bonafide' else 0
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


def load_audio(file_path, sr=16000):
   X, fs = librosa.load(file_path, sr=None)
   if fs != sr:
       X = librosa.resample(X, orig_sr=fs, target_sr=sr)
   return X, sr

def norm_fbank(fbank):
    return fbank
    norm_mean= -8.7086
    norm_std= 4.4163
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

class Dataset_MAE(Dataset):
   def __init__(self, args, list_IDs, labels, base_dir, algo,return_raw_wav=False,balanced_batch=False,task="None",use_DANN=False,is_valid=False,remove_silence=False, cut_set=False, filter_num=250):
       if cut_set:
            # 過濾 list_IDs：只保留編號 ≤ max_num 的樣本
           filtered_list_IDs = []
           for utt_id in list_IDs:
               parts = utt_id.split('_')
               if len(parts) >= 2:
                   try:
                       num = int(parts[1])  # 提取編號（如 p226_261 中的 261）
                       if num <= filter_num:
                           filtered_list_IDs.append(utt_id)
                   except ValueError:
                       print(f"Cannot convert {parts[1]} into int.\n")
           self.list_IDs = filtered_list_IDs 
       else:              
           self.list_IDs = list_IDs
       self.labels_binary = labels[0]
       self.labels_multi_as = labels[1]
       self.labels_multi_ds = labels[2]
       self.labels_multi_qs = labels[3]
       self.base_dir = base_dir
       self.algo = algo
       self.args = args
       self.return_raw_wav=return_raw_wav
       self.cut = 82200
       
       self.task=task
       self.balanced_batch=balanced_batch
       self.use_DANN=use_DANN
       if use_DANN:
           self.COSG_uttid=self.get_COSG_uttid(path="./database/ASVspoof_LA_cm_protocols/dev_maskgct_spoof.txt") if is_valid else self.get_COSG_uttid(path="./database/ASVspoof_LA_cm_protocols/train_maskgct_spoof.txt")
           
       if balanced_batch:
           self.generate_batch()

       self.remove_silence=remove_silence
       self.log="log_len.txt"


   def get_COSG_uttid(self, path):
       with open(path, 'r') as f:
           l_meta = f.readlines()
       file_list=[]
       for line in l_meta:
           key = line.strip().split()[1]
           file_list.append(key)
       return file_list
       
       

   def generate_batch(self):
       # print(self.list_IDs[:10])
       if self.task=="AS":
           LL=self.labels_multi_as
       elif self.task=="DS":
           LL=self.labels_multi_ds
       elif self.task=="QS":
           LL=self.labels_multi_qs
       def split_dict_by_value(LL):
           groups = defaultdict(list)
           for key, value in LL.items():
               groups[value].append(key)
           return [groups[i] for i in sorted(groups.keys())]  # 按 value 升序排列
       self.uttid_split=split_dict_by_value(LL)
       # print(self.uttid_split[0][:10])
       # print(f"Len of dataset:{len(self.uttid_split[0])}")
       
        
    
   def __len__(self):
       return len(self.list_IDs) if self.balanced_batch==False else len(self.uttid_split[0])

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
       if self.balanced_batch==False:
           utt_id = self.list_IDs[index] 
           x_inp = self.read_audio(utt_id)
           binary_target = self.labels_binary[utt_id]
           multi_target_as = self.labels_multi_as[utt_id]
           multi_target_ds = self.labels_multi_ds[utt_id]
           multi_target_qs = self.labels_multi_qs[utt_id]

           if self.use_DANN:
               utt_id_COSG = self.COSG_uttid[index % len(self.COSG_uttid)]
               x2_inp = self.read_audio(utt_id_COSG)
       else:
           utt_id_real = self.uttid_split[0][index]
           utt_ids=[utt_id_real]
           for i in range(1,len(self.uttid_split)):
               utt_ids.append(random.choice(self.uttid_split[i]))

           datas=[]
           labels_b,labels_as,labels_ds,labels_qs=[],[],[],[]
           for utt_id in utt_ids:    
               x_inp = self.read_audio(utt_id)
               datas.append(x_inp)
               labels_b.append(self.labels_binary[utt_id])
               labels_as.append(self.labels_multi_as[utt_id])
               labels_ds.append(self.labels_multi_ds[utt_id])
               labels_qs.append(self.labels_multi_qs[utt_id])
           final_tensor = torch.stack(datas)
           binary_target, multi_target_as,multi_target_ds,multi_target_qs=torch.tensor(labels_b),torch.tensor(labels_as),torch.tensor(labels_ds),torch.tensor(labels_qs)

           if self.use_DANN:
               datas_COSG=[]
               utt_ids_COSG=random.sample(self.COSG_uttid, len(self.uttid_split))
               for utt_id in utt_ids_COSG:    
                   x_inp = self.read_audio(utt_id)
                   datas_COSG.append(x_inp)
               final_tensor_COSG = torch.stack(datas_COSG)

       
       if self.return_raw_wav:
           if self.use_DANN==False:
               return x_inp if self.balanced_batch==False else final_tensor, binary_target, multi_target_as,multi_target_ds,multi_target_qs
           else:
               return (x_inp if self.balanced_batch==False else final_tensor,  x2_inp if self.balanced_batch==False else final_tensor_COSG), binary_target, multi_target_as,multi_target_ds,multi_target_qs
           
       ## ,Shape of x_inp=(82200),  Shape of fbank=(512,128)
       if self.balanced_batch==False:
           fbank = torchaudio.compliance.kaldi.fbank(x_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
           fbank=norm_fbank(fbank)
            
           if self.use_DANN==False:
               return fbank.unsqueeze(0), binary_target, multi_target_as,multi_target_ds,multi_target_qs
           else:
               fbank2 = torchaudio.compliance.kaldi.fbank(x2_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
               fbank2=norm_fbank(fbank2)
               return (fbank.unsqueeze(0), fbank2.unsqueeze(0)), binary_target, multi_target_as,multi_target_ds,multi_target_qs
       else:
           fbanks=[]
           for x_inp in datas:
               fbank = torchaudio.compliance.kaldi.fbank(x_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
               fbank=norm_fbank(fbank)
               fbanks.append(fbank)
           final_tensor = torch.stack(fbanks)
           if self.use_DANN==False:
               return final_tensor, binary_target, multi_target_as,multi_target_ds,multi_target_qs
           else:
               fbanks2=[]
               for x2_inp in datas_COSG:
                   fbank2 = torchaudio.compliance.kaldi.fbank(x2_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
                   fbank2=norm_fbank(fbank2)
                   fbanks2.append(fbank2)
               final_tensor2 = torch.stack(fbanks2)
               return (final_tensor, final_tensor2), binary_target, multi_target_as,multi_target_ds,multi_target_qs
           
class Dataset_MAE_eval(Dataset):
   def __init__(self, list_IDs, base_dir,return_raw_wav=False,remove_silence=False,cut_set=False, filter_num=250):
       if cut_set:
           # 過濾 list_IDs：只保留編號 ≤ max_num 的樣本
           filtered_list_IDs = []
           for utt_id in list_IDs:
               parts = utt_id.split('_')
               if len(parts) >= 2:
                   try:
                       num = int(parts[1])  # 提取編號（如 p226_261 中的 261）
                       if num > filter_num:
                           filtered_list_IDs.append(utt_id)
                   except ValueError:
                       print(f"Cannot convert {parts[1]} into int.\n")
                       
           self.list_IDs = filtered_list_IDs 
       else:    
           self.list_IDs = list_IDs
       self.base_dir = base_dir
       self.cut = 82200
       self.return_raw_wav=return_raw_wav
       self.remove_silence=remove_silence

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
       # x_inp = shuffle_speech(x_inp)
       if self.return_raw_wav:
           return x_inp,utt_id
       fbank = torchaudio.compliance.kaldi.fbank(x_inp.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
       fbank=norm_fbank(fbank)
       return fbank.unsqueeze(0), utt_id


def shuffle_speech(wave_tensor, k=2):
    # wave_tensor shape: (total_length, )
    total_length = wave_tensor.shape[0]
    segment_length = total_length // k

    segments = []
    # 前 k-1 段，每段長度固定為 segment_length
    for j in range(k - 1):
        start = j * segment_length
        end = (j + 1) * segment_length
        segments.append(wave_tensor[start:end])
    # 最後一段為剩餘部分
    segments.append(wave_tensor[(k - 1) * segment_length:])

    # 隨機排列這 k 段
    idx = torch.randperm(k)
    shuffled_segments = [segments[i] for i in idx]

    # 拼接回一個新的 tensor
    return torch.cat(shuffled_segments)




class Dataset_ASVspoof2019_train(Dataset):
   def __init__(self, args, list_IDs, labels, base_dir, algo):
       self.list_IDs = list_IDs
       self.labels_binary = labels[0]
       self.labels_multi_as = labels[1]
       self.labels_multi_ds = labels[2]
       self.labels_multi_qs = labels[3]
       self.base_dir = base_dir
       self.algo = algo
       self.args = args
       self.cut = 64600

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
           raise FileNotFoundError(f"File {utt_id} not found in {self.base_dir}")

       #X, sr = torchaudio.load(file_path)
       X, sr = load_audio(file_path, sr=16000)
       if self.algo:
           X = process_Rawboost_feature(X, sr, self.args, self.algo)
       X_pad = pad(X, self.cut)
       x_inp = Tensor(X_pad)
       binary_target = self.labels_binary[utt_id]
       multi_target_as = self.labels_multi_as[utt_id]
       multi_target_ds = self.labels_multi_ds[utt_id]
       multi_target_qs = self.labels_multi_qs[utt_id]
       return x_inp, binary_target, multi_target_as,multi_target_ds,multi_target_qs

class Dataset_ASVspoof2021_eval(Dataset):
   def __init__(self, list_IDs, base_dir):
       self.list_IDs = list_IDs
       self.base_dir = base_dir
       self.cut = 64600

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
       X_pad = pad(X, self.cut)
       x_inp = torch.FloatTensor(X_pad)
       return x_inp, utt_id


class Dataset_vis(Dataset):
   def __init__(self, list_IDs, base_dir,is_SLM):
       self.list_IDs = list_IDs
       self.base_dir = base_dir
       self.is_SLM=is_SLM
       self.cut = 64600

   def __len__(self):
       return len(self.list_IDs)

   def __getitem__(self, index):
       utt_id = self.list_IDs[index]
       if self.is_SLM:
           codec,key,label=utt_id.split("/")
           utt_id=f"{codec}/{key}"
           if label=="bonafide":
               codec="REAL"
            
       else:
           if len(utt_id)<=8:
               codec="REAL"
           else:
               codec=utt_id[9:]

       
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
       X_pad = pad(X, self.cut)
       x_inp = torch.FloatTensor(X_pad)


       return x_inp, codec

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
