#!/usr/bin/env python3
"""
Batch transcribe audio files using a locally finetuned Whisper model or OpenAI Whisper.

This is a simplified version of batch_transcribe.py that allows you to 
point directly to a local finetuned model path or use standard Whisper models.

Usage:
  # Finetuned model
  python batch_transcribe_finetuned.py --model-path /path/to/model --model-name my_finetuned
  
"""

import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections import defaultdict

import librosa
import pandas as pd
import torch
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Directories
TRANSCRIPTIONS_DIR = os.path.join(os.path.dirname(__file__), 'transcriptions')


def convert_to_wav(input_file):
    """
    Convert audio file to WAV format using ffmpeg.
    Returns the path to the converted file (temporary) or original if already WAV.
    """
    # If already a WAV file, return as-is
    if input_file.lower().endswith('.wav'):
        return input_file, None
    
    # Create a temporary WAV file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    try:
        # Use ffmpeg to convert
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', input_file, '-ar', '16000', '-ac', '1', temp_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        return temp_path, temp_path  # Return path and cleanup path
    except FileNotFoundError:
        os.unlink(temp_path)
        raise RuntimeError("ffmpeg not found. Install it with: sudo apt install ffmpeg")


def load_finetuned_model(model_path):
    """
    Load a locally finetuned Whisper model from the specified path.
    
    Args:
        model_path: Path to the model directory containing config.json, model weights, etc.
    
    Returns:
        Tuple of (model, processor)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Make sure the path is correct and contains model files (config.json, pytorch_model.bin, etc.)"
        )
    
    print(f"  Loading model from: {model_path}")
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Move to best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"  Using device: {device}")
    model = model.to(device)
    
    return model, processor


def load_whisper_model(model_size):
    """
    Load a standard OpenAI Whisper model.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large)
    
    Returns:
        Loaded Whisper model
    """
    print(f"  Loading OpenAI Whisper model: {model_size}")
    
    model = whisper.load_model(model_size)
    
    print(f"  Model loaded successfully")
    
    return model


def transcribe_with_finetuned(audio_file, model, processor):
    """
    Transcribe audio using a locally finetuned HuggingFace Whisper model.
    
    Args:
        audio_file: Path to audio file
        model: Loaded HuggingFace Whisper model
        processor: Whisper processor
    
    Returns:
        Transcription text
    """
    device = next(model.parameters()).device
    
    # Convert to WAV if needed
    wav_path, cleanup_path = convert_to_wav(audio_file)
    
    try:
        # Load and resample audio to 16kHz (Whisper requirement)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            audio, sr = librosa.load(wav_path, sr=16000)
        
        # Process audio
        input_features = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="en",
                task="transcribe",
            )
        
        # Decode the transcription
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    finally:
        # Clean up temporary file if created
        if cleanup_path and os.path.exists(cleanup_path):
            os.unlink(cleanup_path)


def transcribe_with_whisper(audio_file, model, prompt=None):
    """
    Transcribe audio using OpenAI Whisper model.
    
    Args:
        audio_file: Path to audio file
        model: Loaded OpenAI Whisper model
        prompt: Optional initial prompt
    
    Returns:
        Transcription text
    """
    # Convert to WAV if needed 
    wav_path, cleanup_path = convert_to_wav(audio_file)
    
    try:
        # Transcribe with Whisper
        result = model.transcribe(
            wav_path,
            language="en",
            initial_prompt=prompt if prompt else None
        )
        
        return result["text"].strip()
    finally:
        # Clean up temporary file if created
        if cleanup_path and os.path.exists(cleanup_path):
            os.unlink(cleanup_path)


def get_output_filename(prompt, model_name):
    """
    Generate output filename based on the prompt and model name.
    
    Args:
        prompt: Initial prompt used for transcription (or None)
        model_name: Name of the finetuned model
    
    Returns:
        Tuple of (directory_path, filename)
    """
    # Create finetuned models directory
    family_dir = os.path.join(TRANSCRIPTIONS_DIR, 'finetuned')
    os.makedirs(family_dir, exist_ok=True)
    
    if not prompt:
        filename = f"{model_name}_no_prompt.tsv"
    else:
        # Create a short hash of the prompt for unique identification
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
        
        # Also create a sanitized version (first 30 chars, alphanumeric only)
        sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:30].strip('_')
        
        # Combine both for readability and uniqueness
        if sanitized:
            filename = f"{model_name}_{sanitized}_{prompt_hash}.tsv"
        else:
            filename = f"{model_name}_{prompt_hash}.tsv"
    
    return family_dir, filename


def load_all_questions_from_folder(folder_name="extracted_audio_links_with_questions"):
    """
    Load and concatenate all TSV files from the specified folder.
    Returns a dict mapping from URL to (participant_id, question)
    """
    questions_map = {}
    
    # Get the folder path
    folder_path = os.path.join(os.path.dirname(__file__), folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path}/ folder not found.")
        return questions_map
    
    # Find all TSV files in the folder
    tsv_files = sorted(glob.glob(os.path.join(folder_path, "*.tsv")))
    
    if not tsv_files:
        print(f"Warning: No TSV files found in {folder_path}/")
        return questions_map
    
    print(f"Loading questions from {len(tsv_files)} TSV file(s) in {folder_name}/...")
    
    # Load each TSV file and merge into questions_map
    total_loaded = 0
    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            file_count = 0
            for _, row in df.iterrows():
                url = row['url']
                participant_id = row['participant_id']
                question = row['associated_question']
                # Only add if not already present (avoid duplicates)
                if url not in questions_map:
                    questions_map[url] = (participant_id, question)
                    file_count += 1
            
            total_loaded += file_count
            print(f"  {os.path.basename(tsv_file)}: {file_count} questions")
        except Exception as e:
            print(f"  Could not load {os.path.basename(tsv_file)}: {e}")
    
    print(f"Total: {total_loaded} unique questions loaded")
    return questions_map


def get_question_for_file(audio_file, questions_map):
    """
    Find the question associated with an audio file by matching the UUID
    """
    filename = os.path.basename(audio_file)
    
    # Extract UUID from filename
    # Format: participantId_uuid.webm or participantId_uuid_location.webm
    parts = filename.split('_')
    if len(parts) >= 2:
        # Get UUID (second part, strip any extension)
        uuid = parts[1].replace('.webm', '')
        
        # Search for matching URL in questions_map
        for url, (participant_id, question) in questions_map.items():
            if uuid in url:
                return question
    
    return ""


def main():
    parser = argparse.ArgumentParser(
        description='Batch transcribe audio files using a locally finetuned Whisper model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with model path
  python batch_transcribe_finetuned.py --model-path /path/to/model --model-name my_finetuned
  
  # Custom audio directory
  python batch_transcribe_finetuned.py --model-path /path/to/model --model-name my_model --audio-dir custom_audio

Output:
  - Transcriptions are saved to: transcriptions/finetuned/{model_name}_no_prompt.tsv
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the locally finetuned Whisper model directory'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name for the model (used in output filenames and data)'
    )
    
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='audio_files',
        help='Directory containing audio files (default: audio_files)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Audio Transcription - Finetuned Model")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print()
    
    # Load the finetuned model
    print("Loading finetuned model...")
    try:
        model, processor = load_finetuned_model(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    print()
    
    # Check if audio_files directory exists
    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        print(f"Error: {audio_dir}/ directory not found")
        print()
        print("Make sure you've downloaded the audio files first:")
        print("  1. Run: node extract_audio_from_js.js <summary_url>")
        print("  2. Run: ./download_audio.sh")
        print("  3. Then run this script")
        sys.exit(1)
    
    # Load questions from all TSV files in extracted_audio_links_with_questions folder
    questions_map = load_all_questions_from_folder("extracted_audio_links_with_questions")
    print()
    
    # Process URLs from questions_map to create ordered audio file list
    audio_files_ordered = []
    
    if questions_map:
        print(f"Processing {len(questions_map)} URLs to find matching audio files...")
        for url in questions_map.keys():
            # Extract UUID from URL (last part of path before query params)
            if '/vault/' in url:
                # Get the part between /vault/ and the ? or end
                path_part = url.split('/vault/')[1].split('?')[0]
                # Get the last UUID (some paths have multiple UUIDs)
                uuid = path_part.split('/')[-1]
                
                # Strip any file extension (.webm, etc) from the UUID
                if '.' in uuid:
                    uuid = uuid.rsplit('.', 1)[0]
                
                # Find matching audio file with UUID anywhere in filename
                pattern = os.path.join(audio_dir, f'*_{uuid}*.webm')
                matches = glob.glob(pattern)
                if matches:
                    audio_files_ordered.append(matches[0])
        
        print(f"Found {len(audio_files_ordered)} matching audio files")
    else:
        print("Error: No questions loaded from TSV files")
        sys.exit(1)
    
    if audio_files_ordered:
        audio_files = audio_files_ordered
    else:
        print("Error: No matching files from URLs, something has gone wrong")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s)")
    print()
    
    # Generate output file path
    family_dir, filename = get_output_filename(None, args.model_name)
    output_file = os.path.join(family_dir, filename)
    
    print(f"Output directory: {family_dir}/")
    print(f"Output file: {filename}")
    print()
    
    # Check what's already been processed
    processed = set()
    if os.path.exists(output_file):
        print(f"Loading existing transcriptions from {filename}...")
        try:
            existing = pd.read_csv(output_file, sep='\t')
            for _, row in existing.iterrows():
                prompt_used = row.get('prompt', '') if 'prompt' in existing.columns else ''
                if pd.isna(prompt_used):
                    prompt_used = ''
                processed.add((row['participant_id'], row['index'], row['model'], prompt_used))
            print(f"Found {len(processed)} existing transcriptions")
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print("Starting fresh...")
    
    # Create output file with header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("participant_id\tindex\tmodel\tprompt\toriginal_text\ttranscription\n")
        print(f"Created output file: {output_file}")
    elif os.path.getsize(output_file) == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("participant_id\tindex\tmodel\tprompt\toriginal_text\ttranscription\n")
        print(f"Added header to empty output file: {output_file}")
    else:
        print(f"Output file exists, will append new transcriptions")
    
    print()
    print("=" * 60)
    print(f"Starting transcription with {args.model_name}...")
    print("=" * 60)
    print()
    
    # Track participant indices
    participant_indices = defaultdict(int)
    
    success = 0
    failed = 0
    skipped = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        
        # Extract prolific ID from filename (format: participantId_uuid.webm)
        participant_id = filename.split('_')[0]
        
        # Increment index for this participant
        participant_indices[participant_id] += 1
        index = participant_indices[participant_id]
        
        # Get the original text from the questions map
        original_text = get_question_for_file(audio_file, questions_map)
        
        # Check if already processed
        if (participant_id, index, args.model_name, '') in processed:
            skipped += 1
            if i % 50 == 0:  # Only print occasionally to reduce clutter
                print(f"[{i}/{len(audio_files)}] Skipping (already processed)...")
            continue
        
        print(f"[{i}/{len(audio_files)}] Processing: {filename}")
        print(f"  Participant ID: {participant_id} (audio #{index})")
        print(f"  Original text: {original_text[:60]}{'...' if len(original_text) > 60 else ''}")
        
        try:
            # Transcribe with finetuned model
            transcription = transcribe_with_finetuned(audio_file, model, processor)
            
            # Replace tabs and newlines with spaces
            transcription = transcription.replace('\t', ' ').replace('\n', ' ')
            original_text_clean = original_text.replace('\t', ' ').replace('\n', ' ')
            
            # Append to output file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{participant_id}\t{index}\t{args.model_name}\t\t{original_text_clean}\t{transcription}\n")
            
            # Add to processed set
            processed.add((participant_id, index, args.model_name, ''))
            
            print(f"  Transcribed: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
            success += 1
            
        except Exception as e:
            print(f"  Transcription failed: {e}")
            traceback.print_exc()
            failed += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print("TRANSCRIPTION COMPLETE!")
    print("=" * 60)
    print(f"Statistics:")
    print(f"   Success: {success}")
    print(f"   Skipped (already processed): {skipped}")
    print(f"   Failed: {failed}")
    print()
    print(f"Output file: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
