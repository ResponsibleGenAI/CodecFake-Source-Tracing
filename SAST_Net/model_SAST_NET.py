import random
from typing import Union

import numpy as np
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
import matplotlib.pyplot as plt


# from transformers import Wav2Vec2Model, AutoProcessor,Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor
from model_MAE import mae_base
from FusionLayer.transformer_encoder import TransformerEncoder
from fairseq.checkpoint_utils import load_model_ensemble_and_task

from transformers import WhisperFeatureExtractor, WhisperModel

class SSL_Model(nn.Module):
    def __init__(self,device):
        super(SSL_Model, self).__init__()
        
        cp_path = './Pretrain_weight/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = load_model_ensemble_and_task([cp_path])
        self.model = model[0]
#         self.model.eval()
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        # print(emb)
        return emb

class Whisper(nn.Module):
    def __init__(self,device):
        super(Whisper, self).__init__()
        self.processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.model =  WhisperModel.from_pretrained("openai/whisper-small").to(device)
        self.device=device

        # 冻结Whisper
        for param in self.model.parameters():
            param.requires_grad = False
        
    def extract_feat(self, x): #x為形狀是(B,len)的tensor
        batch_features = []
        for i in range(x.shape[0]):
            # 從張量中獲取單個音頻樣本
            audio = x[i].cpu().numpy()
            
            # 使用特徵提取器將音頻轉換為梅爾頻譜圖
            # sampling_rate=16000 表示音頻是以16kHz採樣的
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            batch_features.append(inputs.input_features)
        
        # 合併批次
        input_features = torch.cat(batch_features, dim=0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.encoder(input_features)
            features = outputs.last_hidden_state  # 特征形状: (B, 1500, 768)
            
        features = features[:,:256,:].transpose(1, 2)  # (B, 768, 256)，轉一下
        features = F.avg_pool1d(features, kernel_size=2, stride=2)  # 池化
        features = features.transpose(1, 2) # (B, 128, 768)
        return features
    
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out

class SAST_Net(nn.Module):
    def __init__(self, args, device, task,num_decoder=1,use_SSL_feat=False,mask_2D=False,use_semantic=False,return_emb=False):
        super().__init__()
        self.device = device
        self.num_decoder=num_decoder
        
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]

        # Create SSL model for feature extraction
        self.use_SSL_feat=use_SSL_feat
        self.mask_2D=mask_2D
        if self.use_SSL_feat:
            self.SSL_model=SSL_Model(self.device)
#             for param in self.SSL_model.model.parameters():
#                 param.requires_grad = False
            self.SSL_proj = nn.Linear(1024, 128)
            self.MAE = mae_base(NUM_DECODER=num_decoder,codec_omni_img_shape=(256,128),mask_2d=mask_2D)
        else:
            self.MAE = mae_base(NUM_DECODER=num_decoder,mask_2d=mask_2D)
    
        self.use_semantic=use_semantic
        
        if self.use_semantic:
            self.whisper=Whisper(device) 
            self.ta_transformer=TransformerEncoder(embed_dim=768,
                                                 num_heads=16,
                                                 layers=2,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.1,
                                                 attn_mask=False)
            self.at_transformer=TransformerEncoder(embed_dim=768,
                                                 num_heads=16,
                                                 layers=2,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.1,
                                                 attn_mask=False)
            self.at_ta_transformer=TransformerEncoder(embed_dim=768,
                                                 num_heads=16,
                                                 layers=2,
                                                 attn_dropout=0.0,
                                                 relu_dropout=0.1,
                                                 res_dropout=0.1,
                                                 embed_dropout=0.1,
                                                 attn_mask=False)

        self.LL = nn.Linear(768, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
        )

        if task=="AUX" or task=="VQ":
            num_class=4
        elif task=="DEC":
            num_class=3
        elif task=="Bin":
            num_class=2
            
        # 4-way classification branch  
        self.multiclass_branch = GraphBranch(gat_dims=gat_dims,
                                           pool_ratios=pool_ratios,
                                           temperatures=temperatures,
                                           num_classes=num_class)
        self.return_emb=return_emb

    def wav_to_mel(self,x):
        def norm_fbank(fbank, remove_silence=False):
            norm_mean= -8.7086 if remove_silence==False else -6.1276
            norm_std= 4.4163 if remove_silence==False else 3.8073
            fbank = (fbank - norm_mean) / (norm_std * 2)
            return fbank
        fbanks=[]
        for x_i in x:
            fbank= torchaudio.compliance.kaldi.fbank(x_i.unsqueeze(dim=0), htk_compat=True, sample_frequency=16000, use_energy=False, 
                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            fbanks.append(fbank.unsqueeze(dim=0).unsqueeze(dim=0))
        fbanks=torch.cat(fbanks,dim=0)
        return norm_fbank(fbanks)

    
    def forward(self, x, labels,mask_ratio=0.3):
        if self.use_semantic:
            x_raw=x
            x=self.wav_to_mel(x)
        
        if self.use_SSL_feat:
            self.SSL_model.train()
            x=self.SSL_proj(self.SSL_model.extract_feat(x_raw if self.use_semantic else x)).unsqueeze(0).permute(1, 0, 2, 3) #MAE吃[Batch,1,H,W]的shape
        
        # Extract SSL features
        recon_loss, emb_enc, mask = self.MAE(x,labels,mask_ratio,use_contrasitive=False,use_SSL_feat=self.use_SSL_feat)

        if self.use_semantic:
            sem_feat=self.whisper.extract_feat(x_raw)
            emb_enc_seq_first,sem_feat_seq_first=emb_enc.permute(1, 0, 2), sem_feat.permute(1, 0, 2)
            
            feat_a_to_t = self.at_transformer(emb_enc_seq_first, sem_feat_seq_first, sem_feat_seq_first)
            feat_t_to_a = self.ta_transformer(sem_feat_seq_first, emb_enc_seq_first, emb_enc_seq_first)
            
            x= self.at_ta_transformer(feat_a_to_t, feat_t_to_a, feat_t_to_a)
            x=x.permute(1,0,2)
            x= self.LL(x)
        else:    
            x = self.LL(emb_enc)
   
        def AASIST_forward(x):
            # Shared processing
            x = x.transpose(1, 2)
            x = x.unsqueeze(dim=1)
            x = F.max_pool2d(x, (3, 3))
            x = self.first_bn(x)
            x = self.selu(x)
            
            # Encoder
            x = self.encoder(x)
            x = self.first_bn1(x)
            x = self.selu(x)
            
            # Attention
            w = self.attention(x)
            multiclass_out,last_hidden = self.multiclass_branch(x, w)
            return multiclass_out,last_hidden

        multiclass_out,last_hidden=AASIST_forward(x)
            
        return multiclass_out,recon_loss
        

    def predict(self, x):
        if self.use_semantic:
            x_raw=x
            x=self.wav_to_mel(x)
        
        if self.use_SSL_feat:
            self.SSL_model.eval()
            x=self.SSL_proj(self.SSL_model.extract_feat(x_raw if self.use_semantic else x)).unsqueeze(0).permute(1, 0, 2, 3) #MAE吃[Batch,1,H,W]的shape
        # Extract SSL features
        output= self.MAE.predict(x)
        if self.use_semantic:
            sem_feat=self.whisper.extract_feat(x_raw)            
            emb_enc_seq_first,sem_feat_seq_first=output.permute(1, 0, 2), sem_feat.permute(1, 0, 2)
#             print(emb_enc_seq_first.shape,sem_feat_seq_first.shape)
            feat_a_to_t = self.at_transformer(emb_enc_seq_first, sem_feat_seq_first, sem_feat_seq_first)
            
            feat_t_to_a = self.ta_transformer(sem_feat_seq_first, emb_enc_seq_first, emb_enc_seq_first)
            
            x= self.at_ta_transformer(feat_a_to_t, feat_t_to_a, feat_t_to_a)
            x=x.permute(1,0,2)
            x= self.LL(x)

        else:    
            x = self.LL(output)
        
        # Shared processing
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        
        # Encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        # Attention
        w = self.attention(x)
        
        # Process through both branches
        multiclass_out,last_hidden = self.multiclass_branch(x, w)        

        if self.return_emb:
            return multiclass_out, last_hidden
        else:
            return multiclass_out
    

class GraphBranch(nn.Module):
    def __init__(self, gat_dims, pool_ratios, temperatures, num_classes):
        super().__init__()
        
        self.pos_S = nn.Parameter(torch.randn(1, 42, gat_dims[0]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph attention layers
        self.GAT_layer_S = GraphAttentionLayer(64, gat_dims[0], 
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(64, gat_dims[0], 
                                               temperature=temperatures[1])
        
        # Heterogeneous GAT layers
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        
        # Pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.out_layer = nn.Linear(5 * gat_dims[1], num_classes)
        
    def forward(self, x, w):
        # Spectral attention
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        
        # Temporal attention
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        
        # Graph processing
        gat_S = self.GAT_layer_S(e_S)
        gat_T = self.GAT_layer_T(e_T)
        out_S = self.pool_S(gat_S)
        out_T = self.pool_T(gat_T)
        
        # First inference path
        master1 = self.master1.expand(x.size(0), -1, -1)
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        
        # Second inference path
        master2 = self.master2.expand(x.size(0), -1, -1)
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        
        # Apply dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        
        # Combine features
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        
        # Final readout
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        # print(last_hidden.shape)
        
        return output, last_hidden



def load_XLSR(path,model):
    state_dict = torch.load(path, map_location="cpu")
    model_state_dict = model.state_dict()
    new_dicts={k: model_state_dict[k] for k in model_state_dict.keys()}
    
    for k in new_dicts.keys():
        if k.startswith("SSL_model"):
            k_pron=k.replace("SSL_model","ssl_model")
            new_dicts[k]=state_dict[k_pron]
#         elif k.startswith("multiclass_branch"):
#             k_pron=k.replace("multiclass_branch","multiclass_branch_as")
#             new_dicts[k]=state_dict[k_pron]
        
    model.load_state_dict(new_dicts)
        
    

