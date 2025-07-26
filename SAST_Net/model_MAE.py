from functools import partial
from json import encoder

import torch,os,argparse
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import timm

#from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformerBlock

from AudioMAE.util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from AudioMAE.util.misc import concat_all_gather
from AudioMAE.util.patch_embed import PatchEmbed_new, PatchEmbed_org

from AudioMAE.contrasitive_loss import SupConLoss

class AudioMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, NUM_DECODER=4,img_size=224, patch_size=16, stride=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 audio_exp=True, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, split_pos=False, pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False,
                 energy_weight_scale=2.0, energy_weight_bias=0.5,
                 pretrain_path="./Pretrain_weight/mae_pretrained_base.pth"
                 ):
        super().__init__()
        self.NUM_DECODER=NUM_DECODER
        self.decoder_depth=decoder_depth
        ##audio_exp=True->channel=1, else, channel=3
        self.audio_exp=audio_exp
        self.use_custom_patch=use_custom_patch
        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax=nn.LogSoftmax(dim=-1)

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        print(f"AudioMAE:mask_2D={self.mask_2d}\n")

        self.epoch = epoch

        self.encoder=MAE_Encoder(img_size,patch_size, stride,in_chans,embed_dim,num_heads,mlp_ratio,depth,contextual_depth,norm_layer,use_custom_patch,pos_trainable)
        
        self.all_decoders=nn.ModuleList([MAE_Decoder(embed_dim,decoder_embed_dim,decoder_depth, decoder_num_heads,
                                 self.encoder.patch_embed.num_patches,patch_size,in_chans,decoder_mode,mlp_ratio,
                                 norm_layer,pos_trainable,no_shift,use_custom_patch) for i in range(NUM_DECODER)])
        
        self.proj_k = nn.Linear((patch_size**2),embed_dim)
        self.proj_v = nn.Linear((patch_size**2),embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=16, batch_first=True)
        self.gelu=nn.GELU()
        self.projection_head = ProjectionHead(in_dim=embed_dim, hidden_dim=512, out_dim=128)
#         self.layernorm=nn.LayerNorm(embed_dim)

        self.temp=0.1
        self.margin=0.1
        self.contra_weight=1
        self.CT_loss=SupConLoss(temperature=self.temp)

        self.energy_weight_scale = energy_weight_scale
        self.energy_weight_bias = energy_weight_bias
        
        self.initialize_weights()
        self.load_pretrain(pretrain_path)

    def load_pretrain(self,model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model_state_dict = self.state_dict()
        # print(model_state_dict.keys())
        # 创建一个新的字典来存储匹配的权重
        new_state_dict = {}
        
        for name, param in state_dict["model"].items():
            if name.startswith("blocks") or name.startswith("norm") or name=="cls_token":
                new_state_dict[f'encoder.{name}'] = param
            elif name.startswith("decoder_blocks") or name.startswith("decoder_norm") or name.startswith("decoder_pred") or name=="mask_token":
                for i in range(self.NUM_DECODER):
                    if not (name.endswith("tau") or ("meta_mlp" in name)):
                        if name.startswith("decoder_blocks"):
                            if int(name.split(".")[1])<self.decoder_depth:
                                new_state_dict[f'all_decoders.{i}.{name}'] = param
                        else:
                            new_state_dict[f'all_decoders.{i}.{name}'] = param  
        # 更新当前模型的权重
        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)
        print("Pretrained weights loaded successfully.")     

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.encoder.pos_embed.shape[-1], self.encoder.patch_embed.patch_hw, cls_token=True)    
        else:
            pos_embed = get_2d_sincos_pos_embed(self.encoder.pos_embed.shape[-1], int(self.encoder.patch_embed.num_patches**.5), cls_token=True)
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for d in self.all_decoders:
            if self.audio_exp:   
                decoder_pos_embed = get_2d_sincos_pos_embed_flexible(d.decoder_pos_embed.shape[-1], self.encoder.patch_embed.patch_hw, cls_token=True)
            else:
                decoder_pos_embed = get_2d_sincos_pos_embed(d.decoder_pos_embed.shape[-1], int(self.encoder.patch_embed.num_patches**.5), cls_token=True)
            d.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        for d in self.all_decoders:
            torch.nn.init.normal_(d.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        L = (H/p)*(W/p)
        """
        p = self.encoder.patch_embed.patch_size[0]
        
        if self.audio_exp:
            if self.use_custom_patch: # overlapped patch
                h,w = self.encoder.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
                #x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                #x = torch.einsum('nchpwq->nhwpqc', x)
                #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum('nchpwq->nhwpqc', x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 )
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]    
        h = 1024//p
        w = 128//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs
        
    def reconstruction_loss(self, imgs, pred, mask, norm_pix_loss=False, return_vector=False):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  if return_vector==False else (loss * mask).sum(dim=1) / mask.sum(dim=1)
        return loss      

    def forward(self, imgs, labels, mask_ratio=0.3, use_contrasitive=False, use_SSL_feat=False):
        # 1. 編碼器處理，取得 emb_enc、mask 等資訊
        emb_enc, mask, ids_restore, _ = self.encoder(imgs, mask_ratio, mask_2d=self.mask_2d,use_SSL_feat=use_SSL_feat)
        B = imgs.size(0)

        if self.NUM_DECODER == 1:
            # If there is only one decoder
            decoder = self.all_decoders[0]
            pred, _, _ = decoder(emb_enc, ids_restore)
        
            # Compute reconstruction loss using the predicted output
            reconstruction_loss = self.reconstruction_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        
            # Compute attention output using the prediction as key and value
            attn_output, _ = self.attn(emb_enc, self.proj_k(pred), self.proj_v(pred))
        
        else:
            # If there are multiple decoders, stack the outputs from all decoders along a new dimension
            all_decoder_preds = torch.stack([decoder(emb_enc, ids_restore)[0] for decoder in self.all_decoders], dim=-1)
            
            # Get shape: B=batch size, H=height, W=width, num_decoders=number of decoders
            B, H, W, num_decoders = all_decoder_preds.shape
        
            # Ensure label values are within valid decoder index range
            assert torch.all((labels >= 0) & (labels < num_decoders)), "labels out of index range"
        
            # Create one-hot mask from labels for selecting corresponding decoder outputs
            labels_one_hot = torch.zeros(B, H, W, num_decoders, device=labels.device)
            labels_one_hot.scatter_(dim=3, index=labels.view(B, 1, 1, 1), value=1)
        
            # Select decoder output based on labels
            selected_preds = (all_decoder_preds * labels_one_hot).sum(dim=3)  # Shape: [B, H, W]
        
            # Compute reconstruction loss for each decoder individually
            B, L, C, num_decoders = all_decoder_preds.shape
            losses = []
            for i in range(num_decoders):
                # Extract prediction from the i-th decoder, shape: [B, L, C]
                pred_i = all_decoder_preds[..., i]
                # Compute reconstruction loss for each decoder
                loss_i = self.reconstruction_loss(imgs, pred_i, mask, norm_pix_loss=self.norm_pix_loss, return_vector=True)
                losses.append(loss_i)
        
            # Stack all losses, resulting shape: [B, num_decoders]
            losses = torch.stack(losses, dim=1)
        
            # Select the loss corresponding to the positive (correct) decoder
            loss_pos = torch.gather(losses, dim=1, index=labels.view(B, 1))  # Shape: [B, 1]
        
            # Create index mask for non-target (negative) decoders
            decoder_indices = torch.arange(num_decoders, device=labels.device).unsqueeze(0)  # Shape: [1, num_decoders]
            mask_neg = (decoder_indices != labels.view(B, 1)).float()  # Shape: [B, num_decoders]
        
            # Compute contrastive difference: margin + pos - neg losses
            contrast_diff = self.margin + loss_pos - losses
        
            # Apply ReLU to contrastive difference, mask non-target decoders
            contrast_loss = F.relu(contrast_diff) * mask_neg
        
            # Compute mean contrastive loss over negative decoders
            contrast_loss_batch_mean = contrast_loss.sum(dim=1) / mask_neg.sum(dim=1).clamp(min=1)
        
            # Combine positive decoder loss and weighted contrastive loss
            reconstruction_loss_vec = loss_pos.squeeze() + self.contra_weight * contrast_loss_batch_mean
            reconstruction_loss = reconstruction_loss_vec.mean()
        
            # Compute attention using concatenated outputs from all decoders
            # Move decoder dimension after height: [B, H, D, W]
            all_decoder_preds = all_decoder_preds.permute(0, 1, 3, 2)
        
            # Flatten H and D to form attention input: [B, H * D, W]
            all_decoder_preds_concat = all_decoder_preds.reshape(B, H * num_decoders, W)
        
            # Apply attention using the decoder outputs as key and value
            attn_output, _ = self.attn(emb_enc, self.proj_k(all_decoder_preds_concat), self.proj_v(all_decoder_preds_concat))


        if use_contrasitive:
            contrasitive_loss = 0.1*self.CT_loss(self.projection_head(emb_enc), labels)
            return contrasitive_loss, reconstruction_loss, attn_output, mask
        else:
            return reconstruction_loss, attn_output, mask
    
    def predict(self, imgs):
        emb_enc, mask, ids_restore, _ = self.encoder(imgs, mask_ratio=0.0, mask_2d=False)
        B = imgs.size(0)

        if self.NUM_DECODER==1:
            decoder=self.all_decoders[0]
            pred, _, _ = decoder(emb_enc, ids_restore)
            attn_output, _ = self.attn(emb_enc,self.proj_k(pred),self.proj_v(pred))
        else:
            decoder_preds = []
            for decoder in self.all_decoders:
                pred, _, _ = decoder(emb_enc,ids_restore)
                decoder_preds.append(pred)
            
            all_decoder_preds = torch.stack(decoder_preds, dim=-1)
            B, H, W, num_decoders = all_decoder_preds.shape
            all_decoder_preds = all_decoder_preds.permute(0, 1, 3, 2)
            all_decoder_preds_concat = all_decoder_preds.reshape(B, H * num_decoders, W)
            attn_output, _ = self.attn(emb_enc,self.proj_k(all_decoder_preds_concat),self.proj_v(all_decoder_preds_concat))
    
        return attn_output




class MAE_Encoder(nn.Module):
    def __init__(self,img_size=224,patch_size=16, stride=10,in_chans=3,embed_dim=1024,num_heads=16,mlp_ratio=4.,depth=24,contextual_depth=8,
                 norm_layer=nn.LayerNorm,use_custom_patch=False,pos_trainable=False,):
        super().__init__()
        self.patch_size=patch_size
        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # print(f"x:{x.shape} x_masked:{x_masked.shape} mask:{mask.shape} id_re:{ids_restore.shape}")

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob, use_SSL_feat=False):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch: # overlapped patch
            T=101
            F=12
        else:            
            T=512//self.patch_size if use_SSL_feat==False else 256//self.patch_size
            F=128//self.patch_size
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore

    def forward_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)
        

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        contextual_embs=[]
        for n, blk in enumerate(self.blocks):
            x = blk(x)
            if n > self.contextual_depth:
                contextual_embs.append(self.norm(x))
        #x = self.norm(x)
        contextual_emb = torch.stack(contextual_embs,dim=0).mean(dim=0)

        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)
        #emb = self.encoder_emb(x)

        return contextual_emb

    def forward(self, x, mask_ratio, mask_2d=False, use_SSL_feat=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        

        # masking: length -> length * mask_ratio
        if mask_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=0.5*mask_ratio, mask_f_prob=0.5*mask_ratio,use_SSL_feat=use_SSL_feat)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #emb = self.encoder_emb(x)

        return x, mask, ids_restore, None


class MAE_Decoder(nn.Module):
    def __init__(self,embed_dim=1024,decoder_embed_dim=512,decoder_depth=8, decoder_num_heads=16,
                 num_patches=512,patch_size=16,in_chans=3,decoder_mode=0,mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,pos_trainable=False,no_shift=False,use_custom_patch=False):
        super().__init__()
        self.use_custom_patch=use_custom_patch
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding


        self.no_shift=no_shift


        self.decoder_mode = decoder_mode
        if self.use_custom_patch: # overlapped patches as in AST. Similar performance yet compute heavy
            window_size= (6,6)
            feat_size = (102,12)
        else:
            window_size= (4,4)
            feat_size = (64,8)                
        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0,0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)
                    #shift_size = tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size])
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer, #nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)        
        else:
            # Transfomer
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

    def forward(self, x, ids_restore=None):
        # embed tokens
        x = self.decoder_embed(x)
        
        if ids_restore is not None:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        if self.decoder_mode != 0:
            B,L,D=x.shape
            x = x[:,1:,:]
            if self.use_custom_patch:
                x = x.reshape(B,101,12,D)
                x = torch.cat([x,x[:,-1,:].unsqueeze(1)],dim=1) # hack
                x = x.reshape(B,1224,D)
        if self.decoder_mode > 3: # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B,102,12,256)
                pred = pred[:,:101,:,:]
                pred = pred.reshape(B,1212,256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred, None, None #emb, emb_pixel

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, z):
        proj = self.fc1(z)
        proj = self.gelu(proj)
        proj = self.fc2(proj)
        proj = F.normalize(proj, dim=-1)
        return proj


def mae_base(NUM_DECODER=1,codec_omni_img_shape=(512,128),**kwargs): ##188.24M
    model = AudioMAE(in_chans=1, patch_size=16, img_size=codec_omni_img_shape,NUM_DECODER=NUM_DECODER,
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512,decoder_depth=8,  decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_large(NUM_DECODER=1,**kwargs):
    codec_omni_img_shape=(512,128)
    ####目前只有base的pretrain, 因為embed_dim是768無法匹配，除非embed_dim從1024改768####
    model = AudioMAE(in_chans=1,patch_size=16, img_size=codec_omni_img_shape,NUM_DECODER=NUM_DECODER,
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

        









