# CodecFake Source Tracing

[![Paper](https://img.shields.io/badge/arXiv-2505.12994-b31b1b.svg)](https://arxiv.org/abs/2505.12994)
[![Paper](https://img.shields.io/badge/arXiv-2506.07294-b31b1b.svg)](https://arxiv.org/abs/2506.07294)
[![Paper](https://img.shields.io/badge/arXiv-2501.08238-b31b1b.svg)](https://arxiv.org/abs/2501.08238)
[![Dataset](https://img.shields.io/badge/Dataset-CodecFake+-blue.svg)](https://github.com/ResponsibleGenAI/CodecFake-Plus-Dataset)



**The complete codebase is coming soon!**


## 🛠️ Setup

###  Dataset Download

Download the [CodecFake+ dataset](https://github.com/ResponsibleGenAI/CodecFake-Plus-Dataset) (The dataset is coming soon !)

```
CodecFake+/
├── all_data_16k/          # CoRS + maskgct_vctk set
│   ├── p225_001_audiodec_24k_320d.wav
│   ├── p225_001_bigcodec.wav
│   ├── ....
│   └── s5_400_xocdec_hubert_general_audio.wav
└── SLMdemos_16k/          # CoSG set
    ├── SIMPLESPEECH1/     
    ├── VIOLA/
    ├── ....
    └── MASKGCT/
```

###  Pretrained Weights Setup

- #### For Wav2Vec2-AASIST
    - Place `xlsr2_300m.pt` directly into `w2v2_aasist_baseline/`

- #### For SAST Net
    - Create directory `Pretrain_weight` inside `SAST_Net/`
    - Download and place the following checkpoints in `SAST_Net/Pretrain_weight`:
    
    | Model | Description | Download |
    |:-----:|:-----------:|:--------:|
    | **xlsr2_300m.pt** | Wav2Vec2 pretrained weight | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Pretrain_weight/xlsr2_300m.pt) |
    | **mae_pretrained_base.pth** | AudioMAE pretrained on AudioSet | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Pretrain_weight/mae_pretrained_base.pth) |
    | **tuned_weight.pth** | Wav2Vec2-AASIST on CodecFake+ | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Pretrain_weight/tuned_weight.pth) |

### Environment Setup

```bash
conda env create -f environment.yml
conda activate CodecFakeSourceTracing
```

---

## 🚀 Inference

###  Notation

- #### **Tasks**
    - **BIN**: Binary spoof detection task
    - **VQ**: Vector quantization source tracing task  
    - **AUX**: Auxiliary training objective source tracing task
    - **DEC**: Decoder type source tracing task

- #### **Training Subsets** 
    - **vq**: VQ taxonomy sampling (MVQ : SVQ : SQ = 1:1:1)
    - **aux**: AUX taxonomy sampling (None : Semantic Distillation : Disentanglement = 1:1:1)  
    - **dec**: DEC taxonomy sampling (Time : Freqency = 1:1)

###  Model Checkpoints

- #### **Wav2Vec2-AASIST**

    <details>
    <summary><strong> Single-Task Learning Models</strong></summary>
    
    | Model | Task | Trained Dataset | Download Links |
    |:-----:|:----:|:---------------:|:--------------:|
    | **S_BIN** | BIN | vq / aux / dec | [vq](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/S_BIN_VQ_bal.pth) • [aux](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/S_BIN_AUX_bal.pth) • [dec](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/S_BIN_DEC_bal.pth) |
    | **S_VQ** | VQ | vq | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/S_VQ.pth) |
    | **S_AUX** | AUX | aux | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/S_AUX.pth) |
    | **S_DEC** | DEC | dec | [Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/S_DEC.pth) |
    
    </details>
    
    <details>
    <summary><strong>Dual-Task Learning Models</strong></summary>
    
    | Model | Task | Trained Dataset | Download Links |
    |:-----:|:----:|:---------------:|:--------------:|
    | **D_VQ** | BIN / VQ | vq | [Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/D_VQ.pth) |
    | **D_AUX** | BIN / AUX | aux | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/D_AUX.pth) |
    | **D_DEC** | BIN / DEC | dec | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/D_DEC.pth) |
    
    </details>
    
    <details>
    <summary><strong> Multi-Task Learning Models</strong></summary>
    
    | Model | Task | Trained Dataset | Download Links |
    |:-----:|:----:|:---------------:|:--------------:|
    | **M1** | BIN / VQ / AUX / DEC | vq / aux / dec | [vq](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/M1_VQ_bal.pth) • [aux](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/M1_AUX_bal.pth) • [dec](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/M1_DEC_bal.pth) |
    | **M2** | VQ / AUX / DEC | vq / aux / dec | [vq](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/M2_VQ_bal.pth) • [aux](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/M2_AUX_bal.pth) • [dec](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/Wav2Vec2_baseline/M2_DEC_bal.pth) |
    
    </details>

- #### **SAST Net**

    | Model | Task | Trained Dataset | Download Links |
    |:-----:|:----:|:---------------:|:--------------:|
    | **SAST Net** | BIN | vq / aux / dec | [vq](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/SAST_Net/SAST_Net_BIN_VQ_bal.pth) • [aux](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/SAST_Net/SAST_Net_BIN_AUX_bal.pth) • [dec](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/SAST_Net/SAST_Net_BIN_DEC_bal.pth) |
    | | VQ | vq | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/SAST_Net/SAST_Net_VQ.pth) |
    | | AUX | aux | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/SAST_Net/SAST_Net_AUX.pth) |
    | | DEC | dec | [ Download](https://huggingface.co/CodecFake/CodecFake_Source_Tracing/blob/main/model_checkpoints/SAST_Net/SAST_Net_DEC.pth) |

###  Running Inference

- #### **Wav2Vec2-AASIST**

    ```bash
    cd w2v2_aasist_baseline/
    bash inference.sh ${dataset_type} ${base_dir} ${checkpoint_path} ${model_type}
    ```
    
    **Parameters:**
    - `dataset_type`: `"CoRS"` or `"CoSG"`
    - `base_dir`: Path to dataset directory
      - For CoRS: `"CodecFake+/all_data_16k/"`
      - For CoSG: `"CodecFake+/SLMdemos_16k/"`
    - `checkpoint_path`: Path to model checkpoint
    - `model_type`: `S_BIN` / `S_VQ` / `S_AUX` / `S_DEC` / `D_VQ` / `D_AUX` / `D_DEC` / `M1` / `M2`

- #### **SAST Net**

    ```bash
    cd SAST_Net/
    bash inference.sh ${base_dir} ${dataset_type} ${checkpoint_path} ${task} ${eval_output}
    ```
    
    **Parameters:**
    - `base_dir`: Path to dataset directory
    - `dataset_type`: `"CoRS"` or `"CoSG"`
    - `checkpoint_path`: Path to model checkpoint
    - `task`: `Bin` / `AUX` / `DEC` / `VQ`
    - `eval_output`: Results directory (default: `"./Result"`)

---

## 🎯 Training

- ### **Wav2Vec2-AASIST**

    ```bash
    cd w2v2_aasist_baseline/
    bash train.sh ${base_dir} ${batch_size} ${num_epochs} ${lr} ${model_type} ${sampling_strategy}
    ```

    **Parameters:**
    - `base_dir`: Path to `"CodecFake+/all_data_16k/"`
    - `batch_size`: Batch size (default: `8`)
    - `num_epochs`: Training epochs (default: `20`)
    - `lr`: Learning rate (default: `1e-06`)
    - `model_type`: `S_BIN` / `S_VQ` / `S_AUX` / `S_DEC` / `D_VQ` / `D_AUX` / `D_DEC` / `M1` / `M2`
    - `sampling_strategy`: `VQ` / `AUX` / `DEC`

- ### **SAST Net**

    ```bash
    cd SAST_Net
    bash train.sh ${base_dir} ${save_dir} ${batch_size} ${num_epochs} ${lr} ${task} ${sampling_strategy} ${mask_ratio}
    ```
    
    **Parameters:**
    - `base_dir`: Path to `"CodecFake+/all_data_16k/"`
    - `save_dir`: Checkpoint save directory (default: `./models_SAST_Net`)
    - `batch_size`: Batch size (default: `12`)
    - `num_epochs`: Training epochs (default: `40`)
    - `lr`: Learning rate (default: `1e-05`)
    - `task`: `Bin` / `VQ` / `AUX` / `DEC`
    - `sampling_strategy`: `VQ` / `AUX` / `DEC`
    - `mask_ratio`: MAE mask ratio (default: `0.4`)



## 📚 Citation

If this work helps your research, please consider citing our papers:

```bibtex
@article{chen2025codec,
  title={Codec-Based Deepfake Source Tracing via Neural Audio Codec Taxonomy},
  author={Chen, Xuanjun and Lin, I-Ming and Zhang, Lin and Du, Jiawei and Wu, Haibin and Lee, Hung-yi and Jang, Jyh-Shing Roger Jang},
  journal={arXiv preprint arXiv:2505.12994},
  year={2025}
}

@article{chen2025towards,
  title={Towards Generalized Source Tracing for Codec-Based Deepfake Speech},
  author={Chen, Xuanjun and Lin, I-Ming and Zhang, Lin and Wu, Haibin and Lee, Hung-yi and Jang, Jyh-Shing Roger Jang},
  journal={arXiv preprint arXiv:2506.07294},
  year={2025}
}

@article{chen2025codecfake+,
  title={CodecFake+: A Large-Scale Neural Audio Codec-Based Deepfake Speech Dataset},
  author={Chen, Xuanjun and Du, Jiawei and Wu, Haibin and Zhang, Lin and Lin, I and Chiu, I and Ren, Wenze and Tseng, Yuan and Tsao, Yu and Jang, Jyh-Shing Roger and others},
  journal={arXiv preprint arXiv:2501.08238},
  year={2025}
}
```
