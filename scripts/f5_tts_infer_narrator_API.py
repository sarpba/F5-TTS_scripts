import os
import argparse
import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
import random
import logging
import importlib.util  # Added for dynamic importing
import time  # Added for sleep functionality

import soundfile as sf
import tqdm

from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    # save_spectrogram,  # Eltávolítva
    target_sample_rate,
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything

# Beállítjuk a loggingot
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(processName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("infer_batch.log")
    ]
)
logger = logging.getLogger(__name__)

# Determine the project root directory once, globally
PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger.info(f"Project root determined as: {PROJECT_ROOT}")

class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(vocoder_name, local_path)
        self.load_ema_model(model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, vocoder_name, local_path):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    # def export_spectrogram(self, spect, file_spect):
    #     save_spectrogram(spect, file_spect)  # Eltávolítva

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        file_wave,
        # file_spect=None,  # Eltávolítva
        remove_silence=False,
        speed=1.0,
        nfe_step=32,
        seed=-1,  # módosítva
    ):
        # Validáljuk a speed és nfe_step paramétereket
        if not (0.3 <= speed <= 2.0):
            raise ValueError(f"Invalid speed value: {speed}. Must be between 0.3 and 2.0.")
        if not (16 <= nfe_step <= 64):
            raise ValueError(f"Invalid nfe_step value: {nfe_step}. Must be between 16 and 64.")

        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=logger.info,
            progress=tqdm,
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=nfe_step,
            cfg_strength=2,
            sway_sampling_coef=-1,
            speed=speed,
            fix_duration=None,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        # if file_spect is not None:
        #     self.export_spectrogram(spect, file_spect)  # Eltávolítva

        return wav, sr, spect

def parse_arguments():
    parser = argparse.ArgumentParser(description="F5-TTS Batch Inference Script with Multi-GPU Support")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Input directory containing .wav and corresponding .txt files (reference texts)"
    )
    parser.add_argument(
        "-ig", "--input_gen_dir", type=str, required=True,
        help="Input generation directory containing .txt files (generated texts)"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Output directory to save generated .wav files"
    )
    parser.add_argument(
        "--remove_silence", action="store_true",
        help="Remove silence from generated audio"
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True,
        help="Path to the vocabulary file (e.g., /path/to/vocab.txt)"
    )
    parser.add_argument(
        "--ckpt_file", type=str, required=True,
        help="Path to the model checkpoint file (e.g., /path/to/model.pt)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed of the generated audio (0.3-2.0). Default is 1.0"
    )
    parser.add_argument(
        "--nfe_step", type=int, default=32,
        help="Number of NFE steps (16-64). Default is 32"
    )
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="Maximum number of parallel workers. Defaults to the number of available GPUs."
    )
    parser.add_argument(
        "--norm", type=str, required=False, default=None,
        help="Normalization type (e.g., 'hun', 'eng'). Determines which normalizer to use."
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Random seed for reproducibility. Default is -1, which selects a random seed."
    )
    return parser.parse_args()

def process_files(
    gen_txt_files,
    input_dir,
    input_gen_dir,
    output_dir,
    remove_silence,
    vocab_file,
    ckpt_file,
    speed,
    nfe_step,
    device,
    norm_value,  # Added norm_value parameter
    seed,        # Added seed parameter
    input_pairs,  # Added input_pairs parameter (dict)
    input_pair_list,  # Added input_pair_list parameter (list of tuples)
):
    try:
        # Initialize normalization if norm_value is provided
        if norm_value is not None:
            # **Modified Path Construction Starts Here**
            # Use the globally determined PROJECT_ROOT
            normaliser_path = PROJECT_ROOT / "normalisers" / norm_value / "normaliser.py"

            # Log the path for debugging
            logger.info(f"Normaliser path: {normaliser_path}")

            if not normaliser_path.exists():
                logger.error(f"Normalizer module not found for norm='{norm_value}' at {normaliser_path}")
                sys.exit(1)

            spec = importlib.util.spec_from_file_location("normaliser", normaliser_path)
            normaliser = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(normaliser)

            if not hasattr(normaliser, 'normalize'):
                logger.error(f"The normalizer module '{normaliser_path}' does not have a 'normalize' function.")
                sys.exit(1)

            normalize_fn = normaliser.normalize
            logger.info(f"Loaded normalizer '{norm_value}' from {normaliser_path}")
        else:
            normalize_fn = None
            logger.info("No normalization will be applied as --norm parameter was not provided.")

        # Initialize F5TTS class
        f5tts = F5TTS(
            vocab_file=vocab_file,
            ckpt_file=ckpt_file,
            vocoder_name="vocos",
            device=device
        )
        logger.info(f"Initialized F5TTS on device {device}")

        for gen_txt_path in tqdm.tqdm(gen_txt_files, desc=f"Processing on {device}"):
            # **Rekuzív feldolgozás kezdete**
            relative_path = gen_txt_path.relative_to(input_gen_dir).parent
            output_wav_path = output_dir / relative_path / f"{gen_txt_path.stem}.wav"
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            # **Rekuzív feldolgozás vége**

            # Check if output file already exists
            if output_wav_path.exists():
                logger.info(f"Output file {output_wav_path} already exists. Skipping.")
                continue

            # Determine base name
            base_name = gen_txt_path.stem

            # Check if matching .wav and .txt exist in input_dir
            ref_wav_path = input_dir / f"{base_name}.wav"
            ref_txt_path = input_dir / f"{base_name}.txt"

            if ref_wav_path.exists() and ref_txt_path.exists():
                logger.info(f"Using matching pair for {gen_txt_path.name}: {ref_wav_path.name}, {ref_txt_path.name}")
            else:
                # Randomly select a pair from input_pair_list
                if not input_pair_list:
                    logger.error("No input .wav/.txt pairs available for random assignment.")
                    sys.exit(1)
                selected_pair = random.choice(input_pair_list)
                ref_wav_path, ref_txt_path = selected_pair
                logger.info(f"No matching pair for {gen_txt_path.name}. Randomly selected: {ref_wav_path.name}, {ref_txt_path.name}")

            # Check if reference files exist
            if not ref_wav_path.exists():
                logger.warning(f"Reference wav file {ref_wav_path} does not exist. Skipping {gen_txt_path}.")
                continue

            if not ref_txt_path.exists():
                logger.warning(f"Reference txt file {ref_txt_path} does not exist. Skipping {gen_txt_path}.")
                continue

            # Read the reference text
            with open(ref_txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()

            # Read the generated text
            with open(gen_txt_path, "r", encoding="utf-8") as f:
                gen_text = f.read().strip()

                # Apply normalization to gen_text if normalization is enabled
                if normalize_fn is not None:
                    try:
                        gen_text = normalize_fn(gen_text)
                        logger.debug(f"Normalized gen_text for {gen_txt_path}")
                    except Exception as e:
                        logger.error(f"Normalization failed for {gen_txt_path}: {e}")
                        continue

            # Perform inference
            try:
                f5tts.infer(
                    ref_file=str(ref_wav_path),
                    ref_text=ref_text,
                    gen_text=gen_text,
                    file_wave=str(output_wav_path),
                    # file_spect=None,  # Eltávolítva
                    remove_silence=remove_silence,
                    speed=speed,
                    nfe_step=nfe_step,
                    seed=seed,  # Módosítva
                )
                logger.info(f"Generated audio saved to {output_wav_path}")
            except Exception as e:
                logger.error(f"Error processing {gen_txt_path.relative_to(input_gen_dir)}: {e}", exc_info=True)
                continue

            # **750 ms szünet beiktatása a következő fájl feldolgozása előtt**
            # time.sleep(0.75)
            # logger.debug(f"Paused for 750 ms before processing the next file.")

    except Exception as e:
        logger.critical(f"Critical error in process on device {device}: {e}", exc_info=True)

def main_worker(
    worker_id,
    gen_txt_files_chunk,
    input_dir,
    input_gen_dir,
    output_dir,
    remove_silence,
    vocab_file,
    ckpt_file,
    speed,
    nfe_step,
    norm_value,  # Added norm_value parameter
    seed,        # Added seed parameter
    input_pairs,  # Added input_pairs parameter (dict)
    input_pair_list,  # Added input_pair_list parameter (list of tuples)
):
    # Determine the GPU
    device = f"cuda:{worker_id}"
    logger.info(f"Worker {worker_id} using device {device}")

    process_files(
        gen_txt_files_chunk,
        input_dir,
        input_gen_dir,
        output_dir,
        remove_silence,
        vocab_file,
        ckpt_file,
        speed,
        nfe_step,
        device,
        norm_value,  # Pass norm_value to process_files
        seed,        # Pass seed to process_files
        input_pairs,  # Pass input_pairs to process_files
        input_pair_list,  # Pass input_pair_list to process_files
    )

def main():
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    input_gen_dir = Path(args.input_gen_dir)
    output_dir = Path(args.output_dir)

    # Validate speed and nfe_step parameters
    if not (0.3 <= args.speed <= 2.0):
        logger.error(f"Invalid speed value: {args.speed}. Must be between 0.3 and 2.0.")
        sys.exit(1)
    if not (16 <= args.nfe_step <= 64):
        logger.error(f"Invalid nfe_step value: {args.nfe_step}. Must be between 16 and 64.")
        sys.exit(1)

    # Gather all .wav and .txt files from the input directory recursively
    all_input_wav = list(input_dir.rglob("*.wav"))
    all_input_txt = list(input_dir.rglob("*.txt"))

    if not all_input_wav:
        logger.error(f"No .wav files found in {input_dir} or its subdirectories.")
        sys.exit(1)
    if not all_input_txt:
        logger.error(f"No .txt files found in {input_dir} or its subdirectories.")
        sys.exit(1)

    # Create a dictionary of base_name -> (wav_path, txt_path) if both exist
    input_pairs = {}
    for wav_path in all_input_wav:
        base_name = wav_path.stem
        txt_path = input_dir / f"{base_name}.txt"
        if txt_path.exists():
            input_pairs[base_name] = (wav_path, txt_path)

    if not input_pairs:
        logger.error(f"No matching .wav and .txt pairs found in {input_dir}. Exiting.")
        sys.exit(1)

    # Create a list of all (wav_path, txt_path) tuples for random selection
    input_pair_list = list(input_pairs.values())

    # Gather all generated .txt files from the input_gen_dir recursively
    gen_txt_files = list(input_gen_dir.rglob("*.txt"))

    if not gen_txt_files:
        logger.error(f"No .txt files found in {input_gen_dir} or its subdirectories.")
        sys.exit(1)

    logger.info(f"Number of input pairs: {len(input_pairs)}")
    logger.info(f"Number of generated .txt files: {len(gen_txt_files)}")

    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No GPUs detected. Exiting.")
        sys.exit(1)

    # Determine the number of workers
    max_workers = args.max_workers or num_gpus
    logger.info(f"Number of available GPUs: {num_gpus}")
    logger.info(f"Using {max_workers} parallel workers.")

    # Distribute the gen_txt_files among workers
    chunks = [[] for _ in range(max_workers)]
    for idx, gen_txt_file in enumerate(gen_txt_files):
        chunks[idx % max_workers].append(gen_txt_file)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start the processes
    processes = []
    for worker_id in range(max_workers):
        p = mp.Process(
            target=main_worker,
            args=(
                worker_id % num_gpus,
                chunks[worker_id],
                input_dir,
                input_gen_dir,
                output_dir,
                args.remove_silence,
                args.vocab_file,
                args.ckpt_file,
                args.speed,
                args.nfe_step,
                args.norm,  # Pass norm_value to main_worker
                args.seed,  # Pass seed to main_worker
                input_pairs,  # Pass input_pairs to main_worker
                input_pair_list,  # Pass input_pair_list to main_worker
            )
        )
        p.start()
        processes.append(p)
        logger.info(f"Started process {p.pid} for worker {worker_id} on device cuda:{worker_id % num_gpus}")

    # Wait for all processes to finish
    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.error(f"Process {p.pid} exited with code {p.exitcode}")
        else:
            logger.info(f"Process {p.pid} finished successfully.")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Safer for multi-GPU environments
    main()
