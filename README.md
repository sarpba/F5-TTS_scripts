# F5-TTS_scripts
Prepare F5-TTS anaconda enviroment:
```
conda create -n f5-tts python=3.10 && conda activate f5-tts
conda install git
```
# NVIDIA GPU: install pytorch with your CUDA version, e.g.
```
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
# AMD GPU: install pytorch with your ROCm version, e.g.
```
pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
```
```
pip install git+https://github.com/SWivid/F5-TTS.git

git clone https://github.com/sarpba/F5-TTS_scripts.git
cd F5-TTS_scripts
pip install -r requirements.txt
```

script usage:
```
usage: f5_tts_infer_API.py [-h] -i INPUT_DIR -ig INPUT_GEN_DIR -o OUTPUT_DIR
                           [--remove_silence] --vocab_file VOCAB_FILE
                           --ckpt_file CKPT_FILE [--speed SPEED]
                           [--nfe_step NFE_STEP] [--max_workers MAX_WORKERS]
                           [--norm NORM] [--seed SEED]

F5-TTS Batch Inference Script with Multi-GPU Support

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Input directory containing .wav and corresponding .txt
                        files (reference texts)
  -ig INPUT_GEN_DIR, --input_gen_dir INPUT_GEN_DIR
                        Input generation directory containing .txt files
                        (generated texts) with the same names as input .wav
                        files
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory to save generated .wav files
  --remove_silence      Remove silence from generated audio
  --vocab_file VOCAB_FILE
                        Path to the vocabulary file (e.g., /path/to/vocab.txt)
  --ckpt_file CKPT_FILE
                        Path to the model checkpoint file (e.g.,
                        /path/to/model.pt)
  --speed SPEED         Speed of the generated audio (0.3-2.0). Default is 1.0
  --nfe_step NFE_STEP   Number of NFE steps (16-64). Default is 32
  --max_workers MAX_WORKERS
                        Maximum number of parallel workers. Defaults to the
                        number of available GPUs.
  --norm NORM           Normalization type (e.g., 'hun', 'eng'). Determines
                        which normalizer to use.
  --seed SEED           Random seed for reproducibility. Default is -1, which
                        selects a random seed.
```
