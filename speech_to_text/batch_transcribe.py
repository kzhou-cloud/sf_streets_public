#!/usr/bin/env python3
"""
Batch transcribe audio files and save to a single TSV file
Supports multiple transcription providers: Whisper, Google Cloud Speech-to-Text v2, Deepgram, Phi-4 Multimodal
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict

import librosa
import pandas as pd
import soundfile
import torch
import whisper
from deepgram import DeepgramClient
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Directories
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'whisper_models')
TRANSCRIPTIONS_DIR = os.path.join(os.path.dirname(__file__), 'transcriptions')

# Locally finetuned Whisper models (no PEFT, full model saved locally)
WHISPER_LOCAL_MODELS = {
    'suitcase': {
        'path': os.path.join(os.path.dirname(__file__), '..', 'l2_arctic_finetuning', 'whisper-finetuned-suitcase', 'final'),
        'description': 'Whisper-tiny finetuned on L2-ARCTIC Suitcase Corpus',
    },
}

# Phi-4 Multimodal models
PHI4_MODELS = {
    'phi-4-multimodal': {
        'hf_id': 'microsoft/Phi-4-multimodal-instruct',
        'description': 'Microsoft Phi-4 Multimodal (5.6B) - text, vision, and speech',
    },
}

def get_model_family(model_name):
    """
    Determine which model family a model belongs to
    Returns: 'whisper', 'googlev2', 'deepgram', 'whisper_local', or 'phi4'
    """
    model_lower = model_name.lower()
    if model_lower in ['tiny', 'base', 'small', 'medium', 'large', 'turbo', 'whisper']:
        return 'whisper'
    elif model_lower in ['googlev2', 'chirp_3', 'chirp_2', 'telephony']:
        return 'googlev2'
    elif model_lower in ['deepgram', 'nova-3', 'nova-2', 'enhanced-phonecall', 'enhanced-general', 
                         'base-phonecall', 'base-general', 'nova', 'enhanced', 'base']:
        return 'deepgram'
    elif model_lower in ['whisper_local', 'suitcase']:
        return 'whisper_local'
    elif model_lower in ['phi4', 'phi-4-multimodal', 'phi4-multimodal']:
        return 'phi4'
    else:
        raise ValueError(f"Unknown model: {model_name}")

def transcribe_with_google(audio_file, model_name):
    # Read audio file
    with open(audio_file, 'rb') as audio:
        audio_content = audio.read()
    
    # Initialize client
    client = SpeechClient()
    
    # Get project ID from credentials
    creds_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    with open(creds_path, 'r') as f:
        creds_data = json.load(f)
        project_id = creds_data.get('project_id')
    
    if not project_id:
        raise ValueError("Could not determine project_id from credentials file")
    
    # Map user-friendly names to Google API model names
    # The actual API uses "latest_long", "latest_short", etc.
    # even though the underlying models are Chirp 3, Chirp 2, etc.
    model_map = {
        'chirp_3': 'latest_long',
        'chirp_2': 'latest_short',
        'telephony': 'telephony',
    }
    api_model = model_map.get(model_name, model_name)
    
    # Configure recognition request
    # MUST use global location (other locations don't work)
    recognizer = f"projects/{project_id}/locations/global/recognizers/_"
    
    # Build config - let Google auto-detect the audio format
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model=api_model,
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
    )
    
    # Create the recognition request
    request = cloud_speech.RecognizeRequest(
        recognizer=recognizer,
        config=config,
        content=audio_content,
    )
    
    # Perform the transcription
    response = client.recognize(request=request)
    
    # Extract transcription from response
    transcription_parts = []
    for result in response.results:
        if result.alternatives:
            transcription_parts.append(result.alternatives[0].transcript)
    
    if not transcription_parts:
        return ""
    
    return " ".join(transcription_parts).strip()

def transcribe_with_deepgram(audio_file, model_variant):
    """
    Transcribe audio using Deepgram API
    
    Args:
        audio_file: Path to audio file
        model_variant: Model variant (e.g., 'nova-2', 'nova-3', 'enhanced', 'base', etc.)
                      Note: Flux models are streaming-only and not supported for file transcription
    
    Returns:
        Transcription text
    """

    # Read API key from environment variable
    api_key = os.environ.get('DEEPGRAM_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "DEEPGRAM_API_KEY environment variable is not set.\n"
            "Set it with: export DEEPGRAM_API_KEY='your-api-key'"
        )
    
    # Read audio file
    with open(audio_file, 'rb') as audio:
        audio_content = audio.read()
    
    # STEP 1: Create a Deepgram client
    deepgram: DeepgramClient = DeepgramClient(api_key=api_key)
    
    # STEP 2: Call the transcribe_file method with the audio content and options
    # Note: All file transcriptions use the v1 API endpoint
    # The v2 endpoint is only for streaming/WebSocket connections
    response = deepgram.listen.v1.media.transcribe_file(
        request=audio_content,
        model=model_variant,
        smart_format=True,
    )
    
    # Extract transcription from response
    if (
        response.results
        and response.results.channels
        and len(response.results.channels) > 0
        and response.results.channels[0].alternatives
        and len(response.results.channels[0].alternatives) > 0
    ):
        transcription = response.results.channels[0].alternatives[0].transcript
        return transcription.strip()
    
    return ""


def _patch_phi4_for_generation(model):
    """
    Patch Phi-4 model for compatibility with transformers>=4.50
    where PreTrainedModel no longer inherits from GenerationMixin.
    """
    if not hasattr(model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}
        model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))
    return model


def load_phi4_model(model_name='phi-4-multimodal'):
    """
    Load Microsoft's Phi-4 Multimodal model from HuggingFace
    
    Args:
        model_name: Model variant (currently only 'phi-4-multimodal')
    
    Returns:
        Tuple of (model, processor, generation_config)
    """
    if model_name not in PHI4_MODELS:
        raise ValueError(f"Unknown Phi-4 model: {model_name}. Available: {list(PHI4_MODELS.keys())}")
    
    model_info = PHI4_MODELS[model_name]
    hf_id = model_info['hf_id']
    
    print(f"  Loading Phi-4 model from HuggingFace: {hf_id}")
    print(f"  Description: {model_info['description']}")
    
    # Load processor and model with trust_remote_code for custom model architecture
    processor = AutoProcessor.from_pretrained(
        hf_id,
        trust_remote_code=True
    )
    
    # Determine best available device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16  # MPS doesn't support bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"  Using device: {device}, dtype: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        _attn_implementation="eager",  # Use eager attention for compatibility
    )
    
    # Patch for transformers>=4.50 compatibility
    model = _patch_phi4_for_generation(model)
    
    model = model.to(device)
    
    # Create generation config
    generation_config = GenerationConfig.from_pretrained(hf_id)
    
    print("  Phi-4 Multimodal model loaded successfully")
    
    return model, processor, generation_config


def transcribe_with_phi4(audio_file, model, processor, generation_config, initial_prompt=None):
    """
    Transcribe audio using Microsoft's Phi-4 Multimodal model
    
    Args:
        audio_file: Path to audio file
        model: Loaded Phi-4 model
        processor: Phi-4 processor
        generation_config: Generation configuration
        initial_prompt: Optional additional context prompt
    
    Returns:
        Transcription text
    """
    device = next(model.parameters()).device
    
    # Phi-4 prompt format tokens
    user_prompt_token = '<|user|>'
    assistant_prompt_token = '<|assistant|>'
    prompt_suffix = '<|end|>'
    
    # Build speech transcription prompt
    speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
    if initial_prompt:
        speech_prompt = f"{speech_prompt} {initial_prompt}"
    
    prompt = f'{user_prompt_token}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt_token}'
    
    # Load audio file using librosa (supports webm via ffmpeg fallback)
    # Then create tuple (audio_data, sample_rate) as expected by processor
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        audio_data, sr = librosa.load(audio_file, sr=16000)
    audio = (audio_data, sr)
    
    # Process inputs with audio
    inputs = processor(
        text=prompt,
        audios=[audio],
        return_tensors="pt",
    ).to(device)
    
    # Generate transcription
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            generation_config=generation_config,
            num_logits_to_keep=1,
        )
    
    # Remove input tokens from output
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    
    # Decode the transcription
    transcription = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return transcription.strip()


def get_output_filename(prompt, model_family):
    """
    Generate output filename based on the prompt and model family.
    If no prompt, use default filename.
    If prompt provided, create a sanitized version or hash for the filename.
    
    Args:
        prompt: Initial prompt used for transcription (or None)
        model_family: Model family ('whisper', 'googlev2', 'deepgram')
    
    Returns:
        Tuple of (directory_path, filename)
    """
    # Create model family directory
    family_dir = os.path.join(TRANSCRIPTIONS_DIR, model_family)
    os.makedirs(family_dir, exist_ok=True)
    
    if not prompt:
        filename = "transcriptions_no_prompt.tsv"
    else:
        # Create a short hash of the prompt for unique identification
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
        
        # Also create a sanitized version (first 30 chars, alphanumeric only)
        sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:30].strip('_')
        
        # Combine both for readability and uniqueness
        if sanitized:
            filename = f"transcriptions_{sanitized}_{prompt_hash}.tsv"
        else:
            filename = f"transcriptions_{prompt_hash}.tsv"
    
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
    # Formats:
    #   participantId_uuid.webm
    #   participantId_uuid_location.webm
    #   participantId_uuid.oga_location.webm
    
    # Remove the .webm extension
    name_without_ext = filename
    if name_without_ext.endswith('.webm'):
        name_without_ext = name_without_ext[:-5]
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        # Get everything after the prolific ID (first part)
        # This could be: "uuid", "uuid.oga", "uuid_location", "uuid.oga_location"
        remaining = '_'.join(parts[1:])
        
        # Extract just the UUID using regex (standard UUID format: 8-4-4-4-12 hex chars)
        # This handles all the variations and ignores .oga or location suffixes
        uuid_match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', remaining, re.IGNORECASE)
        if uuid_match:
            uuid_clean = uuid_match.group(1)
            
            # Find matching URL in questions_map
            # The URL format is: .../vault/.../UUID.oga?...
            for url, (participant_id, question) in questions_map.items():
                if uuid_clean in url:
                    return question
    
    return "N/A"

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Batch transcribe audio files using Whisper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (all Whisper model sizes)
  python batch_transcribe.py
  
  # With initial prompt to guide transcription
  python batch_transcribe.py --prompt "This is a recording of street names in San Francisco."
  
  # Use all models in a family
  python batch_transcribe.py --model whisper
  python batch_transcribe.py --model googlev2
  python batch_transcribe.py --model deepgram
  # Use locally finetuned model (Whisper finetuned on Suitcase Corpus)
  python batch_transcribe.py --model suitcase
  
  
Note:
  - Output files are organized by model family in separate directories:
    transcriptions/whisper/
    transcriptions/googlev2/
    transcriptions/deepgram/
    transcriptions/whisper_local/
    transcriptions/phi4/

        """
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Initial prompt to condition the model (guides vocabulary, style, context)'
    )
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        default=['whisper'],
        choices=[
            # Whisper models (individual and all)
            'whisper', 'tiny', 'base', 'small', 'medium', 'large', 'turbo',
            # Google Cloud Speech-to-Text v2 models (individual and all)
            'googlev2', 'chirp_3', 'chirp_2', 'telephony',
            # Deepgram models (individual and all)
            'deepgram', 'nova-3', 'nova-2', 
            'enhanced-phonecall', 'enhanced-general',
            'base-phonecall', 'base-general', 'nova', 'enhanced', 'base',
            # Locally finetuned Whisper models
            'whisper_local', 'suitcase',
            # Microsoft Phi-4 Multimodal
            'phi4', 'phi-4-multimodal',
        ],
        help='Model(s) to use. Whisper: whisper (all) or tiny/base/small/medium/large/turbo. '
                'Google v2: googlev2 (all) or chirp_3/chirp_2/telephony. '
                'Deepgram: deepgram (all) or nova-3/nova-2/enhanced-phonecall/enhanced-general/base-phonecall/base-general. '
                'Local finetuned: whisper_local (all) or suitcase. '
                'Phi-4: phi4 or phi-4-multimodal.'
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Audio Transcription")
    print("=" * 60)
    
    # Determine which models to run
    if 'whisper' in args.model:
        model_list = ['tiny', 'base', 'small', 'medium', 'large']
        print(f"🤖 Models: All Whisper sizes ({', '.join(model_list)})")
    elif 'googlev2' in args.model:
        model_list = ['chirp_3', 'chirp_2', 'telephony']
        print(f"🤖 Models: All Google v2 models ({', '.join(model_list)})")
    elif 'deepgram' in args.model:
        model_list = ['nova-3', 'nova-2', 'enhanced-phonecall', 'enhanced-general', 
                      'base-phonecall', 'base-general']
        print(f"🤖 Models: All Deepgram models ({', '.join(model_list)})")
    elif 'whisper_local' in args.model:
        model_list = list(WHISPER_LOCAL_MODELS.keys())
        print(f"🤖 Models: All locally finetuned Whisper models ({', '.join(model_list)})")
    elif 'phi4' in args.model:
        model_list = list(PHI4_MODELS.keys())
        print(f"🤖 Models: Microsoft Phi-4 Multimodal ({', '.join(model_list)})")
    else:
        model_list = args.model
        print(f"🤖 Model{'s' if len(model_list) > 1 else ''}: {', '.join(model_list)}")
    
    # Group models by family
    models_by_family = defaultdict(list)
    for model in model_list:
        family = get_model_family(model)
        models_by_family[family].append(model)
    
    print(f"Model families: {', '.join(models_by_family.keys())}")
    
    if args.prompt:
        print(f"Initial Prompt: {args.prompt}")
    print()
    
    # Check if audio_files directory exists
    audio_dir = "audio_files"
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
            # Example: https://.../vault/.../05180961-7a85-47d5-af74-a9411c5ea59a?...
            if '/vault/' in url:
                # Get the part between /vault/ and the ? or end
                path_part = url.split('/vault/')[1].split('?')[0]
                # Get the last UUID (some paths have multiple UUIDs)
                uuid = path_part.split('/')[-1]
                
                # Strip any file extension (.oga, .webm, etc) from the UUID
                # Some URLs have .oga but audio files are .webm
                if '.' in uuid:
                    uuid = uuid.rsplit('.', 1)[0]
                
                # Find matching audio file with UUID anywhere in filename
                # Matches: participantId_uuid.webm, participantId_uuid.oga_location.webm, participantId_uuid_location.webm
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
    
    if len(audio_files) == 0:
        print(f"No audio files found in {audio_dir}/")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s)")
    print()
    
    # Process each model family separately
    all_output_files = []
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    for family, family_models in models_by_family.items():
        print("=" * 60)
        print(f"PROCESSING MODEL FAMILY: {family.upper()}")
        print("=" * 60)
        print(f"Models in this family: {', '.join(family_models)}")
        print()
        
        # Generate output file for this model family
        family_dir, filename = get_output_filename(args.prompt, family)
        output_file = os.path.join(family_dir, filename)
        all_output_files.append(output_file)
        
        print(f"Output directory: {family_dir}/")
        print(f"Output file: {filename}")
        if args.prompt:
            print(f"   (Generated from prompt: '{args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}')")
        print()
        
        # Check what's already been processed for this family
        processed = set()
        if os.path.exists(output_file):
            print(f"Loading existing transcriptions from {filename}...")
            try:
                existing = pd.read_csv(output_file, sep='\t')
                for _, row in existing.iterrows():
                    # Handle various formats
                    # Convert NaN to empty string to match how current_prompt is set
                    prompt_used = row.get('prompt', '') if 'prompt' in existing.columns else ''
                    if pd.isna(prompt_used):
                        prompt_used = ''
                    processed.add((row['participant_id'], row['index'], row['model'], prompt_used))
                print(f"Found {len(processed)} existing transcriptions")
            except Exception as e:
                print(f"Warning: Could not load existing file: {e}")
                print("Starting fresh...")
        
        # Create output file with header if it doesn't exist
        # IMPORTANT: Never overwrite existing data - only append or create new
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("participant_id\tindex\tmodel\tprompt\toriginal_text\ttranscription\n")
            print(f"Created output file: {output_file}")
        elif os.path.getsize(output_file) == 0:
            # File exists but is empty - write header
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("participant_id\tindex\tmodel\tprompt\toriginal_text\ttranscription\n")
            print(f"Added header to empty output file: {output_file}")
        else:
            # File exists with data - NEVER modify it, just validate and warn if needed
            try:
                existing = pd.read_csv(output_file, sep='\t', nrows=1)  # Only read header
                missing_cols = []
                expected_cols = ['participant_id', 'index', 'model', 'prompt', 'original_text', 'transcription']
                
                for col in expected_cols:
                    if col not in existing.columns:
                        missing_cols.append(col)
                
                if missing_cols:
                    print(f"WARNING: Output file is missing columns: {', '.join(missing_cols)}")
                    print(f"File will not be modified to preserve existing data.")
                    print(f"New transcriptions will still be appended with all columns.")
                    print(f"You may need to manually update the file later if needed.")
                else:
                    print(f"Output file exists with correct format")
            except Exception as e:
                print(f"Warning: Could not validate file format: {e}")
                print(f"Proceeding with append mode to preserve existing data.")
        
        print()
        print(f"Starting transcription for {family} models...")
        print()
        
        for model_name in family_models:
            print("=" * 60)
            print(f"MODEL: {model_name.upper()}")
            print("=" * 60)
            
            # Load model based on family
            model = None
            processor = None
            generation_config = None
            if family == 'whisper':
                print(f"Loading Whisper {model_name} model...")
                try:
                    model = whisper.load_model(model_name, download_root=MODEL_DIR)
                except Exception as e:
                    print(f"✗ Failed to load {model_name} model: {e}")
                    continue
            elif family == 'googlev2':
                print(f"Initializing Google Cloud Speech-to-Text v2 ({model_name})...")
                # Google models don't need loading, just validation
                print("Google v2 API will be used for transcription")
            elif family == 'deepgram':
                print(f"Initializing Deepgram ({model_name})...")
                # Deepgram models don't need loading, just validation
                print("Deepgram API will be used for transcription")
            elif family == 'phi4':
                print(f"Loading Microsoft Phi-4 Multimodal ({model_name})...")
                try:
                    model, processor, generation_config = load_phi4_model(model_name)
                except Exception as e:
                    print(f"✗ Failed to load {model_name} model: {e}")
                    continue
            
            print()
            
            # Track participant indices for this model
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
                
                # Get current prompt (or empty string if None)
                current_prompt = args.prompt if args.prompt else ''
                
                # Get the original text from the questions map (associated_question column)
                original_text = get_question_for_file(audio_file, questions_map)
                
                # Check if already processed with same prompt
                if (participant_id, index, model_name, current_prompt) in processed:
                    skipped += 1
                    if i % 50 == 0:  # Only print occasionally to reduce clutter
                        print(f"[{i}/{len(audio_files)}] Skipping (already processed)...")
                    continue
                
                print(f"[{i}/{len(audio_files)}] Processing: {filename}")
                print(f"  Participant ID: {participant_id} (audio #{index})")
                print(f"  Original text: {original_text[:60]}{'...' if len(original_text) > 60 else ''}")
                
                try:
                    # Transcribe based on model family
                    if family == 'whisper':
                        # Build transcribe options
                        transcribe_options = {}
                        if args.prompt:
                            transcribe_options['initial_prompt'] = args.prompt
                        
                        # Transcribe with Whisper
                        result = model.transcribe(audio_file, **transcribe_options)
                        transcription = result["text"].strip()
                    
                    elif family == 'googlev2':
                        # Extract variant from model name (e.g., googlev2_long -> long)
                        variant = model_name.replace('googlev2_', '')
                        transcription = transcribe_with_google(audio_file, variant)
                    
                    elif family == 'deepgram':
                        # Model name is already the variant (e.g., nova-2, enhanced-phonecall)
                        transcription = transcribe_with_deepgram(audio_file, model_name)
                    
                    elif family == 'phi4':
                        # Use Phi-4 Multimodal model
                        transcription = transcribe_with_phi4(audio_file, model, processor, generation_config, args.prompt)
                    
                    else:
                        raise ValueError(f"Unknown model family: {family}")
                    
                    # Replace tabs and newlines with spaces
                    transcription = transcription.replace('\t', ' ').replace('\n', ' ')
                    prompt_clean = current_prompt
                    original_text_clean = original_text.replace('\t', ' ').replace('\n', ' ')
                    
                    # Append to output file with original_text column
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(f"{participant_id}\t{index}\t{model_name}\t{prompt_clean}\t{original_text_clean}\t{transcription}\n")
                    
                    # Add to processed set
                    processed.add((participant_id, index, model_name, current_prompt))
                    
                    print(f"  Transcribed successfully")
                    success += 1
                    
                except Exception as e:
                    print(f"  Transcription failed: {e}")
                    traceback.print_exc()
                    failed += 1
                
                print()
            
            # Model summary
            print(f"--- {model_name.upper()} Summary ---")
            print(f"  Success: {success}")
            print(f"  Skipped: {skipped}")
            print(f"  Failed: {failed}")
            print()
            
            total_success += success
            total_skipped += skipped
            total_failed += failed
    
    # Final Summary
    print("=" * 60)
    print("ALL MODELS COMPLETE!")
    print("=" * 60)
    print("Overall Statistics:")
    print(f"   Total files per model: {len(audio_files)}")
    print(f"   Models processed: {len(model_list)}")
    print(f"   Model families: {len(models_by_family)}")
    print(f"   Total transcriptions: {total_success}")
    print(f"   Total skipped: {total_skipped}")
    print(f"   Total failed: {total_failed}")
    print()
    print("Output files:")
    for output_file in all_output_files:
        print(f"   - {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

