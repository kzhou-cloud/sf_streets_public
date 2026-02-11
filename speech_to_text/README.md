# Speech-to-Text

Batch transcription pipeline for evaluating ASR models on street name audio recordings.

## Scripts

- **`batch_transcribe.py`** -- Main transcription script supporting Whisper, Google Cloud Speech-to-Text v2, Deepgram, and Phi-4 Multimodal.
- **`batch_transcribe_finetuned.py`** -- Transcription using locally finetuned Whisper models.
- **`download_models.py`** -- Download Whisper model weights locally.

## Notebooks

- **`transcription_analysis.ipynb`** -- Compare transcription accuracy across models, prompts, and speaker language backgrounds.
- **`transcription_analysis_finetuned.ipynb`** -- Compare finetuned Whisper models against baselines.

## Collecting Audio Recordings

We used [AidaForm](https://aidaform.com/) to collect audio recordings of participants reading street names aloud. The general workflow:

1. **Create a form** on AidaForm with audio recording fields. Each field prompts the participant to say a street name (e.g., "I'm on Cesar Chavez").
2. **Distribute the form** to participants (e.g., via Prolific or other recruitment platforms).
3. **Download responses** from AidaForm's dashboard. Audio recordings are stored as files (`.webm`) linked from each response page. You can extract the audio download URLs from the response pages.
4. **Organize audio files** into `audio_files/` with the naming convention `{participant_id}_{uuid}.webm`.

The transcription scripts expect audio files in `audio_files/` and a set of TSV files in `extracted_audio_links_with_questions/` mapping each audio URL to its participant ID and the associated prompt question.

## Running Transcriptions

```bash
# All Whisper models (tiny through large)
python batch_transcribe.py --model whisper

# With a prompt to guide transcription
python batch_transcribe.py --model whisper --prompt "The user is going to give you their location via a street name."

# Google Cloud Speech-to-Text v2
python batch_transcribe.py --model googlev2

# Deepgram
python batch_transcribe.py --model deepgram

# Finetuned model
python batch_transcribe_finetuned.py --model-path /path/to/model --model-name my_finetuned
```

Output TSV files are organized by model family in `transcriptions/`.
