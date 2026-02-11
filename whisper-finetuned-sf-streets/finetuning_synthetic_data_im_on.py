#!/usr/bin/env python3
"""
Whisper Finetuning with Synthetic Training Data + "I'm on" Prefix

This script:
1. Uses SYNTHETIC audio data (voice cloned) for training
2. PREPENDS "I'm on" audio to TRAINING samples only (NOT test samples)
3. Uses REAL audio data for testing (original participant recordings)
4. Finetunes Whisper to improve transcription for accented speakers

Training data: extracted_validated/*.wav (synthetic voice clones) + im_on.webm prefix
Test data: Original participant recordings from speech_to_text/audio_files (NO prefix)

Usage:
    python finetuning_synthetic_data_im_on.py --model_size large --max_steps 500 --batch_size 8
    
For SLURM:
    sbatch submit_finetuning_synthetic_im_on.sh
"""

import argparse
import evaluate
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Add parent directory to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import normalize_text

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Set up ffmpeg from imageio-ffmpeg (bundled binary, no system install needed)
FFMPEG_PATH = None
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
    # Also set for pydub/audioread
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass  # Fall back to system ffmpeg

# Global constants
SAMPLE_RATE = 16000

# Global variable to store the "I'm on" audio prefix
IM_ON_AUDIO = None

# Participants excluded from test set for data quality reasons
EXCLUDED_PARTICIPANTS = [
    # Add participant IDs here as needed
]

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Whisper on synthetic SF Streets audio data with 'I'm on' prefix"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_size", 
        type=str, 
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to finetune"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint. Can be a path to checkpoint dir, a run name (e.g., 'whisper_large_all_20260128_123456'), or 'latest' to find the most recent checkpoint."
    )
    # Data configuration
    parser.add_argument(
        "--im_on_audio",
        type=str,
        default=None,
        help="Path to 'I'm on' audio file (default: ../im_on.webm)"
    )
    parser.add_argument(
        "--synthetic_project_dir",
        type=str,
        default="cloned_sf_streets",
        help="Name of the voice cloning project directory (default: cloned_sf_streets)"
    )
    parser.add_argument(
        "--synthetic_data_dir",
        type=str,
        default=None,
        help="Full path to synthetic data directory (overrides --synthetic_project_dir if provided)"
    )
    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default=None,
        help="Path to original audio files directory (for test data)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter synthetic training data to a specific language code (e.g., 'hi', 'zh-cn', 'ar'). If not specified, uses all languages."
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=50,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=-1,
        help="Maximum training steps (overrides num_train_epochs if > 0)"
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=25,
        help="Evaluation frequency (steps)"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=50,
        help="Checkpoint save frequency (steps)"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=25,
        help="Logging frequency (steps)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None,
        help="Name for this run (used in output directory)"
    )
    
    # Weights & Biases
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="whisper-sf-streets-synthetic-im-on",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team or username)"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=928745,
        help="Random seed for reproducibility"
    )
    
    # Evaluation options
    parser.add_argument(
        "--skip_baseline_eval", 
        action="store_true",
        help="Skip baseline evaluation before training"
    )
    parser.add_argument(
        "--n_test_samples", 
        type=int, 
        default=50,
        help="Number of samples for detailed before/after comparison"
    )
    
    # Early stopping
    parser.add_argument(
        "--early_stopping_loss_threshold",
        type=float,
        default=0.01,
        help="Stop training if loss falls below this threshold"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help="Number of eval steps with no improvement before stopping"
    )
    
    return parser.parse_args()


# =============================================================================
# SETUP & UTILITIES
# =============================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / "training.log"
    
    # Create logger
    logger = logging.getLogger("whisper_finetuning_synthetic_im_on")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        n_gpus = torch.cuda.device_count()
        return device, f"CUDA ({device_name}, {n_gpus} GPU(s))"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon)"
    else:
        return torch.device("cpu"), "CPU"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_cache_key(
    synthetic_dir: Path,
    audio_dir: Path,
    im_on_audio: str,
    language: str,
    n_train_samples: int,
    n_test_samples: int
) -> str:
    """Generate a cache key based on data configuration.
    
    This ensures that cache is automatically invalidated when:
    - Different data directories are used
    - Language filter changes
    - Different im_on audio file is used
    - Number of samples changes
    
    Args:
        synthetic_dir: Path to synthetic audio directory
        audio_dir: Path to original audio directory
        im_on_audio: Path to im_on audio file
        language: Language filter (or "all")
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
    
    Returns:
        Short hash string for cache directory naming
    """
    # Create a string combining all cache-relevant parameters
    cache_params = f"{synthetic_dir}|{audio_dir}|{im_on_audio}|{language}|{n_train_samples}|{n_test_samples}"
    
    # Hash it to get a short, filesystem-safe identifier
    cache_hash = hashlib.md5(cache_params.encode()).hexdigest()[:12]
    
    return cache_hash


# =============================================================================
# SYNTHETIC DATA LOADING
# =============================================================================
def parse_synthetic_filename(filename: str) -> Dict[str, str]:
    """Parse synthetic audio filename to extract metadata.
    
    Filename format: {speaker_id}_{street_name}_{language_code}.wav
    Examples:
        - 001_alemany_hi.wav -> speaker=001, street=alemany, lang=hi
        - 002_bay_shore_ne-NP.wav -> speaker=002, street=bay_shore, lang=ne-NP
        - 003_diamond_heights_vi.wav -> speaker=003, street=diamond_heights, lang=vi
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 3:
        return None
    
    speaker_id = parts[0]
    
    # Language code is the last part
    lang_code = parts[-1]

    # Street name is everything in between
    street_name = '_'.join(parts[1:-1])
    
    return {
        'speaker_id': speaker_id,
        'street_name': street_name,
        'language_code': lang_code,
    }


def load_synthetic_training_data(synthetic_dir: Path, logger: logging.Logger, 
                                  samples_per_lang_street: int = 2, seed: int = 479) -> pd.DataFrame:
    """Load synthetic audio files as training data.
    
    Args:
        synthetic_dir: Path to synthetic audio files
        logger: Logger instance
        samples_per_lang_street: Max samples per language per street (default: 2)
        seed: Random seed for reproducible sampling
    
    Returns DataFrame with columns: speaker_id, street_name, audio_path, transcription
    """
    train_data_synthetic_files = list(synthetic_dir.glob("*.wav"))
    logger.info(f"Found {len(train_data_synthetic_files)} synthetic audio files in {synthetic_dir}")
    
    train_data = []
    for audio_file in train_data_synthetic_files:
        parsed = parse_synthetic_filename(audio_file.name)
        if parsed is None:
            logger.warning(f"Could not parse filename: {audio_file.name}")
            continue
        
        # Construct transcription with "i'm on" prefix to match real audio format
        street_readable = parsed['street_name'].replace('_', ' ')
        transcription = f"i'm on {street_readable}"
        
        train_data.append({
            'speaker_id': parsed['speaker_id'],
            'language_code': parsed['language_code'],
            'street_name': transcription,  # Full transcription for compatibility
            'audio_path': str(audio_file),
            'transcription': transcription,
        })
    
    train_df = pd.DataFrame(train_data)
    logger.info(f"Loaded {len(train_df)} total synthetic samples")
    
    if len(train_df) > 0:
        logger.info(f"  Unique speakers: {train_df['speaker_id'].nunique()}")
        logger.info(f"  Unique streets: {train_df['street_name'].nunique()}")
        logger.info(f"  Languages: {train_df['language_code'].unique().tolist()}")
        
        # Sample N datapoints per language per street
        if samples_per_lang_street > 0:
            train_df = train_df.groupby(['language_code', 'street_name'], group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), samples_per_lang_street), random_state=seed)
            ).reset_index(drop=True)
            logger.info(f"Sampled {samples_per_lang_street} per language per street: {len(train_df)} samples")
    else:
        logger.warning("No synthetic samples found! Check that the directory exists and contains .wav files.")
    
    return train_df


# =============================================================================
# ORIGINAL DATA LOADING (for test set)
# =============================================================================

def create_audio_file_map(audio_dir: Path, logger: logging.Logger = None) -> Dict:
    """Create mapping from (participant_id, street_name) -> audio file path.
    
    Filters out participants with known data quality issues.
    """
    test_data_audio_files = list(audio_dir.glob("*.webm"))
    audio_file_map = {}
    excluded_count = 0
    
    for audio_file in test_data_audio_files:
        filename = audio_file.stem
        parts = filename.split('_')
        if len(parts) >= 3:
            participant_id = parts[0]
            
            # Skip excluded participants
            if participant_id in EXCLUDED_PARTICIPANTS:
                excluded_count += 1
                continue
            
            street_parts = parts[2:]
            street_name = '_'.join(street_parts)
            audio_file_map[(participant_id, street_name)] = audio_file
    
    if logger and excluded_count > 0:
        logger.info(f"Excluded {excluded_count} audio files from {len(EXCLUDED_PARTICIPANTS)} bad participants")
    
    return audio_file_map


def normalize_street_for_comparison(street: str) -> str:
    """Normalize street name for comparison by removing apostrophes and extra spaces.
    
    Handles naming inconsistencies between data sources:
    - Synthetic audio: O_SHAUGHNESSY, HUNTERS_POINT (uppercase with underscores)
    - Original audio: oshaughnessy, hunters_point (lowercase with underscores)
    """
    normalized = street.lower().replace("'", "").replace("'", "").replace(" ", "_")
    
    # Handle specific naming inconsistencies (AFTER apostrophe removal)
    if normalized == "bayshore":
        normalized = "bay_shore"
    if normalized == "o_shaughnessy":
        normalized = "oshaughnessy"  # Match audio_files format
    
    return normalized


def create_test_split(
    audio_file_map: Dict,
    selected_streets: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """Create test split using original audio data 
    
    Test set uses selected streets from all participants.
    Ground truth is derived from audio filenames.
    """
    # Convert streets to clean names for file matching
    selected_streets_clean = set()
    for street in selected_streets:
        street_clean = normalize_street_for_comparison(street.replace("i'm on ", ""))
        selected_streets_clean.add(street_clean)
    
    # Build test set: selected streets from ALL participants
    test_data = []
    
    for (pid, street_clean), audio_path in audio_file_map.items():
        # Normalize the street name from the audio file for comparison
        street_clean_normalized = normalize_street_for_comparison(street_clean)
        if street_clean_normalized not in selected_streets_clean:
            continue
        
        # Construct ground truth transcription directly from filename
        transcription = f"i'm on {street_clean.replace('_', ' ')}"
        
        test_data.append({
            'participant_id': pid,
            'street_name': transcription,
            'audio_path': str(audio_path),
            'transcription': transcription.lower()
        })
    
    test_df = pd.DataFrame(test_data)
    logger.info(f"Test set: {len(test_df)} samples")
    if len(test_df) > 0:
        logger.info(f"Test set participants: {test_df['participant_id'].nunique()}")
        logger.info(f"Test set unique streets: {test_df['street_name'].nunique()}")
    
    return test_df


# =============================================================================
# AUDIO LOADING WITH "I'M ON" PREFIX
# =============================================================================

def load_im_on_audio(im_on_path: str, logger: logging.Logger) -> np.ndarray:
    """Load the 'I'm on' audio prefix once at the start."""
    global IM_ON_AUDIO
    
    logger.info(f"Loading 'I'm on' audio from: {im_on_path}")
    IM_ON_AUDIO = load_and_resample_audio(im_on_path, SAMPLE_RATE)
    logger.info(f"'I'm on' audio loaded: {len(IM_ON_AUDIO)} samples ({len(IM_ON_AUDIO)/SAMPLE_RATE:.2f} seconds)")
    
    return IM_ON_AUDIO


def load_and_resample_audio(audio_path: str, target_sr: int = 16000):
    """Load audio file and resample to target sample rate.
    
    Uses ffmpeg directly for formats librosa can't handle (webm, etc.).
    """
    # First try librosa directly (works for wav, flac, etc.)
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio
    except Exception:
        pass  # Fall through to ffmpeg
    
    # Use ffmpeg to convert to wav, then load with librosa
    ffmpeg_cmd = FFMPEG_PATH if FFMPEG_PATH else "ffmpeg"
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
        tmp_path = tmp_file.name
        
        # Convert to wav using ffmpeg
        cmd = [
            ffmpeg_cmd,
            "-i", audio_path,
            "-ar", str(target_sr),  # Resample
            "-ac", "1",              # Mono
            "-f", "wav",             # Output format
            "-y",                    # Overwrite
            tmp_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        
        # Load the converted wav file
        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        return audio


def prepare_dataset(df: pd.DataFrame, logger: logging.Logger, id_column: str = 'participant_id', add_im_on_prefix: bool = True):
    """Convert DataFrame to HuggingFace Dataset with audio.
    
    Args:
        df: DataFrame with audio paths and transcriptions
        logger: Logger instance
        id_column: Column name for speaker/participant ID
        add_im_on_prefix: If True, prepends "I'm on" audio to each sample
    """
    if add_im_on_prefix and IM_ON_AUDIO is None:
        raise RuntimeError("IM_ON_AUDIO not loaded! Call load_im_on_audio() first.")
    
    audio_arrays = []
    valid_indices = []
    
    desc = "Loading audio (with 'I'm on' prefix)" if add_im_on_prefix else "Loading audio"
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            # Load the main audio
            audio = load_and_resample_audio(row['audio_path'], SAMPLE_RATE)
            
            # Conditionally prepend "I'm on" audio
            if add_im_on_prefix:
                combined_audio = np.concatenate([IM_ON_AUDIO, audio])
            else:
                combined_audio = audio
            
            audio_arrays.append(combined_audio)
            valid_indices.append(idx)
        except Exception as e:
            logger.error(f"Error loading {row['audio_path']}: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    dataset_dict = {
        'audio': [{'array': audio, 'sampling_rate': SAMPLE_RATE} for audio in audio_arrays],
        'transcription': df.loc[valid_indices, 'transcription'].tolist(),
        'speaker_id': df.loc[valid_indices, id_column].tolist(),
    }
    
    return Dataset.from_dict(dataset_dict)


def prepare_features(batch, processor):
    """Prepare features for Whisper model."""
    audio = batch["audio"]
    
    # Use numpy arrays for proper dataset storage (not PyTorch tensors)
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np"
    ).input_features[0]
    
    labels = processor.tokenizer(batch["transcription"]).input_ids
    
    # Return a new dict with all columns to ensure they're properly added
    return {
        "input_features": input_features,
        "labels": labels,
        "speaker_id": batch["speaker_id"],
        "transcription": batch["transcription"],
    }


# =============================================================================
# DATA COLLATOR & METRICS
# =============================================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper finetuning."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def create_compute_metrics(processor, wer_metric, normalize_text):
    """Create compute_metrics function with processor and normalize_text in closure."""
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize predictions and labels using utils.normalize_text
        # (handles lowercasing, accent removal, spacing, spelling corrections)
        pred_str_normalized = [normalize_text(p) for p in pred_str]
        label_str_normalized = [normalize_text(l) for l in label_str]
        
        wer = 100 * wer_metric.compute(predictions=pred_str_normalized, references=label_str_normalized)
        return {"wer": wer}
    
    return compute_metrics


# =============================================================================
# EARLY STOPPING CALLBACK
# =============================================================================

class EarlyStoppingOnLowLossCallback(TrainerCallback):
    """Stop training when loss falls below a threshold or WER stops improving."""
    
    def __init__(self, loss_threshold: float = 0.01, patience: int = 3, logger=None):
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.logger = logger
        self.best_wer = float('inf')
        self.patience_counter = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check training loss after each logging step."""
        if logs is None:
            return
        
        # Check if training loss is very low
        train_loss = logs.get('loss')
        if train_loss is not None and train_loss < self.loss_threshold:
            if self.logger:
                self.logger.info(f"Early stopping: training loss ({train_loss:.6f}) below threshold ({self.loss_threshold})")
            control.should_training_stop = True
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check WER improvement after evaluation."""
        if metrics is None:
            return
        
        current_wer = metrics.get('eval_wer')
        if current_wer is not None:
            if current_wer < self.best_wer:
                self.best_wer = current_wer
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.logger:
                    self.logger.info(f"No WER improvement for {self.patience_counter}/{self.patience} evals (best: {self.best_wer:.2f}%, current: {current_wer:.2f}%)")
                
                if self.patience_counter >= self.patience:
                    if self.logger:
                        self.logger.info(f"Early stopping: no WER improvement for {self.patience} evaluations")
                    control.should_training_stop = True


# =============================================================================
# EVALUATION & COMPARISON
# =============================================================================
def transcribe_audio(audio_path: str, model, processor, device):
    """Transcribe audio using model and processor (without any prefix added)."""
    # Load audio WITHOUT prepending I'm on audio
    audio_array = load_and_resample_audio(audio_path, target_sr=16000)
    
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def run_comparison_evaluation(
    test_df: pd.DataFrame,
    original_model,
    original_processor,
    finetuned_model,
    finetuned_processor,
    device,
    normalize_text,
    n_samples: int,
    logger: logging.Logger,
    output_dir: Path,
    use_wandb: bool = False
):
    """Run before/after comparison on test samples."""
    logger.info(f"\nRunning detailed comparison on {n_samples} test samples...")
    
    n_samples = min(n_samples, len(test_df))
    sample_test = test_df.sample(n=n_samples, random_state=42)
    
    results = []
    original_correct = 0
    finetuned_correct = 0
    
    for idx, row in tqdm(sample_test.iterrows(), total=n_samples, desc="Comparing"):
        audio_path = row['audio_path']
        ground_truth = row['transcription']
        
        try:
            orig_result = transcribe_audio(audio_path, original_model, original_processor, device).strip().lower()
            fine_result = transcribe_audio(audio_path, finetuned_model, finetuned_processor, device).strip().lower()
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {type(e).__name__}: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            continue
        
        orig_normalized = normalize_text(orig_result)
        fine_normalized = normalize_text(fine_result)
        truth_normalized = normalize_text(ground_truth)
        
        orig_match = orig_normalized == truth_normalized
        fine_match = fine_normalized == truth_normalized
        
        original_correct += orig_match
        finetuned_correct += fine_match
        
        status = "IMPROVED" if fine_match and not orig_match else \
                 "REGRESSION" if orig_match and not fine_match else \
                 "BOTH_CORRECT" if orig_match and fine_match else "BOTH_WRONG"
        
        results.append({
            'participant_id': row['participant_id'],
            'ground_truth': ground_truth,
            'original_transcription': orig_result,
            'finetuned_transcription': fine_result,
            'original_correct': orig_match,
            'finetuned_correct': fine_match,
            'status': status
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate actual number of processed samples (some may have been skipped due to errors)
    n_processed = len(results_df)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Samples processed: {n_processed}/{n_samples}")
    logger.info(f"Original model accuracy:  {original_correct}/{n_processed} ({100*original_correct/n_processed:.1f}%)")
    logger.info(f"Finetuned model accuracy: {finetuned_correct}/{n_processed} ({100*finetuned_correct/n_processed:.1f}%)")
    logger.info(f"Improvement: {finetuned_correct - original_correct} more correct")
    
    # Count status categories
    status_counts = results_df['status'].value_counts()
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    # Log comparison results to wandb
    if use_wandb:
        # Log summary metrics (use actual processed count)
        wandb.run.summary["comparison_original_accuracy"] = original_correct / n_processed
        wandb.run.summary["comparison_finetuned_accuracy"] = finetuned_correct / n_processed
        wandb.run.summary["comparison_improved_count"] = status_counts.get("IMPROVED", 0)
        wandb.run.summary["comparison_regression_count"] = status_counts.get("REGRESSION", 0)
        wandb.run.summary["comparison_n_processed"] = n_processed
        wandb.run.summary["comparison_n_requested"] = n_samples
        
        # Log comparison table
        comparison_table = wandb.Table(
            columns=["participant_id", "ground_truth", "original", "finetuned", "status"],
            data=[
                [row['participant_id'], row['ground_truth'], 
                 row['original_transcription'], row['finetuned_transcription'], 
                 row['status']]
                for _, row in results_df.iterrows()
            ]
        )
        wandb.log({"comparison_results": comparison_table})
    
    # Save detailed results
    results_df.to_csv(output_dir / "comparison_results.csv", index=False)
    logger.info(f"Saved comparison results to {output_dir / 'comparison_results.csv'}")
    
    return results_df


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def main():
    args = parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    
    # Set default im_on audio path
    if args.im_on_audio is None:
        args.im_on_audio = str(script_dir.parent / "im_on.webm")
    
    # Create run name and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lang_suffix = f"_{args.language}" if args.language else "_all"
    run_name = args.run_name or f"whisper_{args.model_size}_synthetic_im_on{lang_suffix}_{timestamp}"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "runs" / run_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("WHISPER FINETUNING WITH SYNTHETIC DATA + 'I'M ON' PREFIX")
    logger.info("=" * 60)
    
    # Initialize Weights & Biases (only on main process for distributed training)
    # Check if we're the main process (rank 0) or running single-process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    if not args.no_wandb and is_main_process:
        try:
            # Load API key from WANDB_API_KEY environment variable
            if not os.environ.get("WANDB_API_KEY"):
                logger.warning("WANDB_API_KEY not set. Set it with: export WANDB_API_KEY='your-key'")
            
            tags = [f"whisper-{args.model_size}", "synthetic_data", "voice_cloned", "im_on_prefix"]
            if args.language and args.language.lower() != "all":
                tags.append(f"lang-{args.language}")
            else:
                tags.append("lang-all")
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                dir=str(output_dir),
                tags=tags,
            )
            entity_str = f"{args.wandb_entity}/" if args.wandb_entity else ""
            logger.info(f"Weights & Biases initialized: {entity_str}{args.wandb_project}/{run_name}")
            use_wandb = True
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False
    elif not is_main_process:
        use_wandb = False  # Non-main processes don't use wandb
    else:
        use_wandb = False
        logger.info("Weights & Biases disabled")
    
    # Log configuration
    config = vars(args)
    config['output_dir'] = str(output_dir)
    config['run_name'] = run_name
    config['timestamp'] = timestamp
    config['data_type'] = 'synthetic_im_on'
    
    logger.info("\nConfiguration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"\nRandom seed: {args.seed}")
    
    # Get device
    device, device_info = get_device()
    logger.info(f"Device: {device_info}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Load "I'm on" audio ONCE at the start
    logger.info("\n" + "=" * 60)
    logger.info("LOADING 'I'M ON' AUDIO PREFIX")
    logger.info("=" * 60)
    load_im_on_audio(args.im_on_audio, logger)
    
    # Set data directories
    if args.synthetic_data_dir:
        # Full path specified - use it directly
        synthetic_dir = Path(args.synthetic_data_dir)
    else:
        # Use project directory name to construct path
        synthetic_dir = script_dir.parent / "voice_cloning" / args.synthetic_project_dir / "extracted_validated"
    
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
    else:
        audio_dir = script_dir.parent / "speech_to_text" / "audio_files"
    
    logger.info(f"Synthetic data directory: {synthetic_dir}")
    logger.info(f"Original audio directory: {audio_dir}")
    
    # ==========================================================================
    # LOAD TRAINING DATA (SYNTHETIC)
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING SYNTHETIC TRAINING DATA")
    logger.info("=" * 60)
    
    train_df = load_synthetic_training_data(synthetic_dir, logger, samples_per_lang_street=2, seed=args.seed)
    
    # Check if we have any training data
    if len(train_df) == 0:
        logger.error("No training data found! Check that:")
        logger.error(f"  1. Synthetic directory exists: {synthetic_dir}")
        logger.error(f"  2. Directory contains .wav files")
        sys.exit(1)
    
    # Filter by language if specified (None or "all" means use all languages)
    if args.language and args.language.lower() != "all" and len(train_df) > 0:
        original_count = len(train_df)
        available_langs = train_df['language_code'].unique().tolist()
        train_df = train_df[train_df['language_code'] == args.language]
        logger.info(f"Filtered to language '{args.language}': {len(train_df)}/{original_count} samples")
        if len(train_df) == 0:
            logger.error(f"No samples found for language '{args.language}'. Available: {available_langs}")
            sys.exit(1)
    elif args.language and args.language.lower() == "all":
        logger.info(f"Using all languages: {len(train_df)} samples")
    
    # Normalize street names to lowercase for consistency
    if len(train_df) > 0:
        train_df['street_name'] = train_df['street_name'].str.lower()
        train_df['transcription'] = train_df['transcription'].str.lower()
    
    # ==========================================================================
    # LOAD TEST DATA (ORIGINAL AUDIO FILES)
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING ORIGINAL TEST DATA")
    logger.info("=" * 60)
    
    # Create audio file mapping (filters out bad participants)
    audio_file_map = create_audio_file_map(audio_dir, logger)
    logger.info(f"Mapped {len(audio_file_map)} audio files")
    
    # Get participant stats
    unique_pids = set(pid for (pid, _) in audio_file_map.keys())
    n_participants = len(unique_pids)
    logger.info(f"Total participants: {n_participants}")
    
    # Debug: Check which streets are available in audio files
    audio_streets = set()
    for (pid, street_name) in audio_file_map.keys():
        audio_streets.add(normalize_street_for_comparison(street_name))
    logger.info(f"Unique streets in audio files (normalized): {sorted(audio_streets)}")
    logger.info(f"Number of unique streets in audio files: {len(audio_streets)}")
    
    # ==========================================================================
    # TRAIN/TEST DATA SETUP
    # - Training: Synthetic data (overlapping street names allowed)
    # - Test: Original audio files (all available streets)
    # ==========================================================================
    
    # Use all available streets from audio files for testing
    test_streets = [f"i'm on {s.replace('_', ' ')}" for s in audio_streets]
    
    logger.info(f"\nTraining data: {len(train_df)} samples")
    logger.info(f"Test data: using {len(test_streets)} streets from original audio")
    
    # Create test split (no CSV needed!)
    test_df = create_test_split(audio_file_map, test_streets, logger)
    
    # Log overlap information
    if len(test_df) > 0:
        test_streets_in_data = {normalize_street_for_comparison(s.replace("i'm on ", "")) 
                                for s in test_df['street_name'].unique()}
        train_streets_in_data = {normalize_street_for_comparison(x.replace("i'm on ", ""))
                                for x in train_df['street_name'].unique()}
        overlapping_streets = test_streets_in_data.intersection(train_streets_in_data)
        
        logger.info(f"\nStreet name overlap between train and test:")
        logger.info(f"  Train streets: {len(train_streets_in_data)}")
        logger.info(f"  Test streets: {len(test_streets_in_data)}")
        logger.info(f"  Overlapping streets: {len(overlapping_streets)}")
        logger.info(f"  Overlap percentage: {100 * len(overlapping_streets) / len(test_streets_in_data):.1f}%")
    
    # Save train/test splits (train_df is now filtered to only include train streets)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    logger.info(f"Saved train/test splits to {output_dir}")
    
    # Log to wandb
    if use_wandb:
        wandb.config.update({
            "n_total_streets": len(test_streets),
            "n_total_participants": n_participants,
            "n_synthetic_train_samples": len(train_df),
            "n_real_test_samples": len(test_df),
            "im_on_audio_path": args.im_on_audio,
        }, allow_val_change=True)
    
    # ==========================================================================
    # LOAD MODEL & PROCESSOR
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING MODEL")
    logger.info("=" * 60)
    
    # Resolve checkpoint path if resuming
    resume_checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_arg = args.resume_from_checkpoint
        
        if checkpoint_arg == "latest":
            # Find the most recent run with checkpoints
            runs_dir = script_dir / "runs"
            if runs_dir.exists():
                run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                                  key=lambda x: x.stat().st_mtime, reverse=True)
                for run_dir in run_dirs:
                    checkpoints_dir = run_dir / "checkpoints"
                    if checkpoints_dir.exists():
                        checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() 
                                                  if d.is_dir() and d.name.startswith("checkpoint-")])
                        if checkpoint_dirs:
                            resume_checkpoint_path = str(checkpoint_dirs[-1])
                            logger.info(f"Found latest checkpoint: {resume_checkpoint_path}")
                            break
        elif Path(checkpoint_arg).exists():
            # Direct path to checkpoint
            resume_checkpoint_path = checkpoint_arg
        else:
            # Assume it's a run name - look for it in runs/
            run_dir = script_dir / "runs" / checkpoint_arg / "checkpoints"
            if run_dir.exists():
                checkpoint_dirs = sorted([d for d in run_dir.iterdir() 
                                          if d.is_dir() and d.name.startswith("checkpoint-")])
                if checkpoint_dirs:
                    resume_checkpoint_path = str(checkpoint_dirs[-1])
                    logger.info(f"Found checkpoint for run '{checkpoint_arg}': {resume_checkpoint_path}")
        
        if resume_checkpoint_path:
            logger.info(f"Will resume training from: {resume_checkpoint_path}")
        else:
            logger.warning(f"Could not find checkpoint for: {checkpoint_arg}")
    
    model_name = f"openai/whisper-{args.model_size}"
    
    # Load processor from base model (doesn't change during training)
    logger.info(f"Loading processor from: {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    
    # Load model from checkpoint or base model
    if resume_checkpoint_path:
        logger.info(f"Loading model weights from checkpoint: {resume_checkpoint_path}...")
        model = WhisperForConditionalGeneration.from_pretrained(resume_checkpoint_path)
    else:
        logger.info(f"Loading base model: {model_name}...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Set language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.generation_config.suppress_tokens = []
    
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # ==========================================================================
    # PREPARE DATASETS
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING DATASETS")
    logger.info("=" * 60)
    
    # Generate cache key based on data configuration
    # This automatically invalidates cache when data/config changes
    cache_key = generate_cache_key(
        synthetic_dir=synthetic_dir,
        audio_dir=audio_dir,
        im_on_audio=args.im_on_audio,
        language=args.language if args.language else "all",
        n_train_samples=len(train_df),
        n_test_samples=len(test_df)
    )
    
    # Use cache key in directory names to auto-invalidate when parameters change
    train_cache_path = output_dir / f"cached_train_dataset_{cache_key}"
    test_cache_path = output_dir / f"cached_test_dataset_{cache_key}"
    
    logger.info(f"Cache key: {cache_key}")
    
    if train_cache_path.exists() and test_cache_path.exists():
        logger.info("Loading cached datasets (validated by cache key)...")
        train_dataset = load_from_disk(str(train_cache_path))
        test_dataset = load_from_disk(str(test_cache_path))
        logger.info(f"Loaded cached train dataset: {len(train_dataset)} samples")
        logger.info(f"Loaded cached test dataset: {len(test_dataset)} samples")
    else:
        logger.info("Building new datasets (no valid cache found)...")
        
        logger.info("Preparing synthetic training dataset (with 'I'm on' prefix)...")
        train_dataset = prepare_dataset(train_df, logger, id_column='speaker_id', add_im_on_prefix=True)
        logger.info(f"Training samples: {len(train_dataset)}")
        
        logger.info("Preparing test dataset (original audio WITHOUT 'I'm on' prefix)...")
        test_dataset = prepare_dataset(test_df, logger, id_column='participant_id', add_im_on_prefix=False)
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Apply feature preprocessing
        logger.info("Preprocessing features...")
        train_dataset = train_dataset.map(
            lambda x: prepare_features(x, processor),
            remove_columns=["audio"]
        )
        test_dataset = test_dataset.map(
            lambda x: prepare_features(x, processor),
            remove_columns=["audio"]
        )
        
        # Save cached datasets for future runs
        logger.info("Saving cached datasets...")
        train_dataset.save_to_disk(str(train_cache_path))
        test_dataset.save_to_disk(str(test_cache_path))
        logger.info(f"Cached datasets saved to {output_dir}")
        logger.info(f"  Train cache: {train_cache_path.name}")
        logger.info(f"  Test cache: {test_cache_path.name}")
    
    # ==========================================================================
    # SETUP TRAINING
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SETTING UP TRAINING")
    logger.info("=" * 60)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    compute_metrics = create_compute_metrics(processor, wer_metric, normalize_text)
    
    # Use max_steps if specified (> 0), otherwise use num_train_epochs
    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Mixed precision: accelerate's --mixed_precision flag overrides this when using multi-GPU
        "fp16": torch.cuda.is_available(),
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "logging_dir": str(output_dir / "logs"),
        "report_to": ["wandb", "tensorboard"] if use_wandb else ["tensorboard"],
        "run_name": run_name,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "push_to_hub": False,
        "predict_with_generate": True,
        "generation_max_length": 225,
        "save_total_limit": 3,  # Keep only last 3 checkpoints
        "remove_unused_columns": False,  # Keep all columns, let data collator handle selection
    }
    
    # Use max_steps if specified, otherwise use num_train_epochs
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
        logger.info(f"Training with max_steps={args.max_steps}")
    else:
        training_kwargs["num_train_epochs"] = args.num_train_epochs
        logger.info(f"Training for {args.num_train_epochs} epochs")
    
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingOnLowLossCallback(
        loss_threshold=args.early_stopping_loss_threshold,
        patience=args.early_stopping_patience,
        logger=logger
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[early_stopping_callback],
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Warmup steps: {args.warmup_steps}")
    if args.max_steps > 0:
        logger.info(f"  Max steps: {args.max_steps}")
    else:
        logger.info(f"  Num train epochs: {args.num_train_epochs}")
    logger.info(f"  Early stopping loss threshold: {args.early_stopping_loss_threshold}")
    logger.info(f"  Early stopping patience: {args.early_stopping_patience} evals")
    
    # ==========================================================================
    # BASELINE EVALUATION
    # ==========================================================================
    if not args.skip_baseline_eval:
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE EVALUATION")
        logger.info("=" * 60)
        
        logger.info("Evaluating baseline model (before finetuning)...")
        baseline_results = trainer.evaluate()
        baseline_wer = baseline_results['eval_wer']
        logger.info(f"Baseline WER: {baseline_wer:.2f}%")
        
        # Log baseline to wandb
        if use_wandb:
            wandb.log({"baseline_wer": baseline_wer})
            wandb.run.summary["baseline_wer"] = baseline_wer
        
        # Save baseline results
        with open(output_dir / "baseline_results.json", 'w') as f:
            json.dump(baseline_results, f, indent=2)
    else:
        baseline_wer = None
        logger.info("Skipping baseline evaluation")
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    
    if resume_checkpoint_path:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint_path}")
    else:
        logger.info("Starting finetuning on synthetic data with 'I'm on' prefix...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint_path)
    
    logger.info("\nTraining complete!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    
    # Save training metrics
    train_metrics = train_result.metrics
    train_metrics['train_loss'] = train_result.training_loss
    with open(output_dir / "training_metrics.json", 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # ==========================================================================
    # FINAL EVALUATION
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    
    logger.info("Evaluating finetuned model on original test data...")
    finetuned_results = trainer.evaluate()
    finetuned_wer = finetuned_results['eval_wer']
    
    logger.info("\nRESULTS COMPARISON")
    logger.info("=" * 40)
    if baseline_wer is not None:
        logger.info(f"Baseline WER:   {baseline_wer:.2f}%")
    logger.info(f"Finetuned WER:  {finetuned_wer:.2f}%")
    if baseline_wer is not None:
        improvement = baseline_wer - finetuned_wer
        logger.info(f"Improvement:    {improvement:.2f}%")
    
    # Log final results to wandb
    if use_wandb:
        wandb.run.summary["finetuned_wer"] = finetuned_wer
        if baseline_wer is not None:
            wandb.run.summary["wer_improvement"] = baseline_wer - finetuned_wer
            wandb.run.summary["wer_improvement_pct"] = ((baseline_wer - finetuned_wer) / baseline_wer) * 100
    
    # Save finetuned results
    with open(output_dir / "finetuned_results.json", 'w') as f:
        json.dump(finetuned_results, f, indent=2)
    
    # ==========================================================================
    # SAVE FINAL MODEL (only on main process)
    # ==========================================================================
    final_model_dir = output_dir / "final_model"
    
    if is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("SAVING MODEL")
        logger.info("=" * 60)
        
        trainer.save_model(str(final_model_dir))
        processor.save_pretrained(str(final_model_dir))
        logger.info(f"Model saved to {final_model_dir}")
    
    # Wait for main process to finish saving before comparison
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # ==========================================================================
    # DETAILED COMPARISON (only on main process)
    # ==========================================================================
    if is_main_process and args.n_test_samples > 0:
        logger.info("\n" + "=" * 60)
        logger.info("BEFORE/AFTER COMPARISON")
        logger.info("=" * 60)
        
        # Load original model for comparison
        logger.info("Loading original model for comparison...")
        original_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        original_processor = WhisperProcessor.from_pretrained(model_name)
        
        # Load finetuned model
        logger.info("Loading finetuned model...")
        finetuned_model = WhisperForConditionalGeneration.from_pretrained(str(final_model_dir)).to(device)
        finetuned_processor = WhisperProcessor.from_pretrained(model_name)  # Load from base model (processor doesn't change)
        
        run_comparison_evaluation(
            test_df,
            original_model,
            original_processor,
            finetuned_model,
            finetuned_processor,
            device,
            normalize_text,
            args.n_test_samples,
            logger,
            output_dir,
            use_wandb=use_wandb
        )
    
    # ==========================================================================
    # SUMMARY (only on main process)
    # ==========================================================================
    if is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        summary = {
            'run_name': run_name,
            'model_size': args.model_size,
            'data_type': 'synthetic_im_on',
            'n_train_samples': len(train_df),
            'n_test_samples': len(test_df),
            'baseline_wer': baseline_wer,
            'finetuned_wer': finetuned_wer,
            'improvement': baseline_wer - finetuned_wer if baseline_wer else None,
            'training_loss': train_result.training_loss,
            'output_dir': str(output_dir),
            'im_on_audio_path': args.im_on_audio,
        }
        
        # Log summary to wandb
        if use_wandb:
            wandb.run.summary["n_train_samples"] = len(train_df)
            wandb.run.summary["n_test_samples"] = len(test_df)
            wandb.run.summary["training_loss"] = train_result.training_loss
            wandb.run.summary["model_size"] = args.model_size
            wandb.run.summary["data_type"] = "synthetic_im_on"
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Run name: {run_name}")
        logger.info(f"Training samples (synthetic): {len(train_df)}")
        logger.info(f"Test samples (original): {len(test_df)}")
        if baseline_wer is not None:
            logger.info(f"Baseline WER: {baseline_wer:.2f}%")
        logger.info(f"Finetuned WER: {finetuned_wer:.2f}%")
        if baseline_wer is not None:
            logger.info(f"Improvement: {baseline_wer - finetuned_wer:.2f}%")
        logger.info(f"\nAll outputs saved to: {output_dir}")
        
        # Finish wandb run
        if use_wandb:
            wandb.finish()
            logger.info("Weights & Biases run finished")
        
        logger.info("\nFINISHED!")


if __name__ == "__main__":
    main()

