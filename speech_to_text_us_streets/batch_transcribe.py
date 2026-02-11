#!/usr/bin/env python3
"""
Batch transcribe audio files using Whisper and save to a single TSV file
"""

import argparse
import glob
import os
import sys
import traceback
import warnings

import pandas as pd
import whisper

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'whisper_models')
TRANSCRIPTIONS_DIR = os.path.join(os.path.dirname(__file__), 'transcriptions')

# Available Whisper model sizes
WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']

# Output file path
OUTPUT_FILE = os.path.join(TRANSCRIPTIONS_DIR, 'whisper', 'transcriptions.tsv')


def load_all_questions_from_folder(folder_name="extracted_audio_links_with_questions"):
    """
    Load all question TSV files from a folder and build a mapping from audio URL UUID to question.
    
    The TSV files have columns: url, participant_id, associated_question
    The audio files are named: participantId_uuid_streetname.webm (or formId_participantId_uuid_streetname.webm)
    
    Returns:
        dict: Mapping from UUID to question text
    """
    questions_map = {}
    
    folder_path = os.path.join(os.path.dirname(__file__), folder_name)
    
    if not os.path.exists(folder_path):
        print(f"  Warning: Questions folder not found: {folder_path}")
        return questions_map
    
    tsv_files = glob.glob(os.path.join(folder_path, "*.tsv"))
    
    if not tsv_files:
        print(f"  Warning: No TSV files found in {folder_path}")
        return questions_map
    
    print(f"  Loading questions from {len(tsv_files)} TSV file(s)...")
    
    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            
            for _, row in df.iterrows():
                url = row.get('url', '')
                question = row.get('associated_question', '')
                
                if url and question:
                    # Extract UUID from URL (last path segment before query params)
                    url_path = url.split('?')[0]
                    uuid = url_path.split('/')[-1]
                    
                    if uuid and len(uuid) > 20:  # Valid UUID-like string
                        questions_map[uuid] = question
        except Exception as e:
            print(f"  Warning: Error loading {tsv_file}: {e}")
    
    print(f"  Loaded {len(questions_map)} question mappings")
    return questions_map


def get_question_for_file(audio_file, questions_map):
    """
    Get the original question/text for an audio file.
    
    Audio files are named: participantId_uuid_streetname.webm (or formId_participantId_uuid_streetname.webm)
    The UUID is extracted and matched against the questions map.
    
    Args:
        audio_file: Path to audio file
        questions_map: Dict mapping UUID to question text
    
    Returns:
        Question text or empty string if not found
    """
    filename = os.path.basename(audio_file)
    
    # Try to extract UUID from filename
    # Format can be: participantId_uuid_streetname.webm or formId_participantId_uuid_streetname.webm
    parts = filename.replace('.webm', '').split('_')
    
    # Look for a UUID-like part (36 chars with hyphens)
    for part in parts:
        if len(part) == 36 and part.count('-') == 4:
            if part in questions_map:
                return questions_map[part]
    
    return ""


def main():
    parser = argparse.ArgumentParser(
        description='Batch transcribe audio files using Whisper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe with all Whisper model sizes (default)
  python batch_transcribe.py
  
  # Use specific model size(s)
  python batch_transcribe.py --model base
  python batch_transcribe.py --model small medium large

Note:
  - Output file: transcriptions/whisper/transcriptions.tsv
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        default=['whisper'],
        choices=['whisper'] + WHISPER_MODELS,
        help='Model(s) to use. "whisper" runs all sizes, or specify: tiny/base/small/medium/large/turbo'
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Audio Transcription (Whisper)")
    print("=" * 60)
    
    # Determine which models to run
    if 'whisper' in args.model:
        model_list = WHISPER_MODELS.copy()
        print(f"🤖 Models: All Whisper sizes ({', '.join(model_list)})")
    else:
        model_list = args.model
        print(f"🤖 Model{'s' if len(model_list) > 1 else ''}: {', '.join(model_list)}")
    
    print()
    
    # Check if audio_files directory exists
    audio_dir = "audio_files"
    if not os.path.isdir(audio_dir):
        print(f"❌ Error: {audio_dir}/ directory not found")
        print()
        print("Make sure you've downloaded the audio files first:")
        print("  1. Run: node extract_audio_from_js.js <aidaform_url>")
        print("  2. Run: ./download_audio.sh")
        sys.exit(1)
    
    # Find audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.webm")))
    
    if not audio_files:
        print(f"❌ Error: No .webm files found in {audio_dir}/")
        sys.exit(1)
    
    print(f"📁 Found {len(audio_files)} audio files in {audio_dir}/")
    print()
    
    # Load questions from TSV files
    print("📋 Loading original questions...")
    questions_map = load_all_questions_from_folder()
    print()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Load existing transcriptions to allow resumption
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, sep='\t')
            for _, row in existing_df.iterrows():
                key = (row['participant_id'], row['audio_index'], row['model'])
                processed.add(key)
            print(f"📝 Found {len(processed)} existing transcriptions in {OUTPUT_FILE}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing transcriptions: {e}")
    else:
        # Create new file with header
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("participant_id\taudio_index\tmodel\toriginal_text\ttranscription\n")
        print(f"📝 Created new output file: {OUTPUT_FILE}")
    
    print()
    
    # Track statistics
    total_success = 0
    total_skipped = 0
    total_failed = 0
    
    # Process each model
    for model_name in model_list:
        print("=" * 60)
        print(f"🔊 Loading Whisper model: {model_name}")
        print("=" * 60)
        
        try:
            model = whisper.load_model(model_name, download_root=MODEL_DIR)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            continue
        
        print()
        
        success = 0
        skipped = 0
        failed = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            filename = os.path.basename(audio_file)
            
            # Parse filename to extract participant_id and index
            # Format: formId_participantId_uuid_streetname.webm or participantId_uuid_streetname.webm
            parts = filename.replace('.webm', '').split('_')
            
            # Try to identify the participant_id (24-char hex string)
            participant_id = None
            for part in parts:
                if len(part) == 24 and all(c in '0123456789abcdef' for c in part):
                    participant_id = part
                    break
            
            if not participant_id:
                participant_id = parts[0] if parts else 'unknown'
            
            # Get audio index for this participant
            participant_files = [f for f in audio_files if participant_id in f]
            try:
                index = participant_files.index(audio_file) + 1
            except ValueError:
                index = 1
            
            # Get original question text
            original_text = get_question_for_file(audio_file, questions_map)
            
            # Check if already processed
            if (participant_id, index, model_name) in processed:
                skipped += 1
                if i % 50 == 0:  # Only print occasionally to reduce clutter
                    print(f"[{i}/{len(audio_files)}] Skipping (already processed)...")
                continue
            
            print(f"[{i}/{len(audio_files)}] Processing: {filename}")
            print(f"  Participant ID: {participant_id} (audio #{index})")
            print(f"  Original text: {original_text[:60]}{'...' if len(original_text) > 60 else ''}")
            
            try:
                # Transcribe with Whisper
                result = model.transcribe(audio_file)
                transcription = result["text"].strip()
                
                # Replace tabs and newlines with spaces
                transcription = transcription.replace('\t', ' ').replace('\n', ' ')
                original_text_clean = original_text.replace('\t', ' ').replace('\n', ' ')
                
                # Append to output file
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{participant_id}\t{index}\t{model_name}\t{original_text_clean}\t{transcription}\n")
                
                # Add to processed set
                processed.add((participant_id, index, model_name))
                
                print(f"  ✓ Transcribed successfully")
                success += 1
                
            except Exception as e:
                print(f"  ✗ Transcription failed: {e}")
                traceback.print_exc()
                failed += 1
            
            print()
        
        # Model summary
        print(f"--- {model_name.upper()} Summary ---")
        print(f"  ✓ Success: {success}")
        print(f"  ⊘ Skipped: {skipped}")
        print(f"  ✗ Failed: {failed}")
        print()
        
        total_success += success
        total_skipped += skipped
        total_failed += failed
    
    # Final Summary
    print("=" * 60)
    print("✅ ALL MODELS COMPLETE!")
    print("=" * 60)
    print("📊 Overall Statistics:")
    print(f"   Total files per model: {len(audio_files)}")
    print(f"   Models processed: {len(model_list)}")
    print(f"   ✓ Total transcriptions: {total_success}")
    print(f"   ⊘ Total skipped: {total_skipped}")
    print(f"   ✗ Total failed: {total_failed}")
    print()
    print(f"📁 Output file: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
