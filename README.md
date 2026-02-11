# Speech-to-Text Evaluation for Street Names

[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude-D97757?logo=anthropic&logoColor=white)](https://claude.ai)

Evaluating speech-to-text accuracy for U.S. street names across speakers with diverse language backgrounds.

## Overview

This project investigates how well ASR systems transcribe street names spoken by people with different native languages. It includes:

- **Multi-model comparison**: Whisper (multiple sizes), Google Cloud Speech-to-Text, Deepgram, and Phi-4
- **Finetuning pipeline**: Whisper finetuned on synthetic voice-cloned street name audio
- **Voice cloning**: Synthetic training data generation using Coqui XTTS-v2
- **Street name dataset**: Extraction and linguistic classification of U.S. street names
- **Analysis**: Performance evaluation by speaker language background

## Repository Structure

```
├── speech_to_text/                # Core transcription pipeline & analysis
│   ├── batch_transcribe.py        # Multi-model transcription (Whisper, Google, Deepgram, Phi-4)
│   ├── batch_transcribe_finetuned.py
│   ├── transcription_analysis.ipynb
│   └── transcription_analysis_finetuned.ipynb
│
├── speech_to_text_us_streets/     # Extension to US-wide street names
│   ├── batch_transcribe.py
│   └── transcription_analysis_us.ipynb
│
├── voice_cloning/                 # Synthetic data generation with XTTS-v2
│   ├── voice_cloning.ipynb        # SF street names
│   └── voice_cloning_us_streets.ipynb
│
├── whisper-finetuned-sf-streets/  # Whisper finetuning
│   ├── finetuning_synthetic_data_im_on.py
│   └── finetuning_synthetic_data_im_on_us_streets.py
│
├── street_names_dataset/          # Street name extraction & classification
│   ├── 1_extract_street_names.ipynb
│   ├── 2_classifying_street_names.ipynb
│   ├── 3_analysis.ipynb
│   └── data/                      # City-level street name data (see data/README.md)
│
├── google_places/                 # Location validation via Google Places API
│   └── google_places_api.ipynb
│
└── utils.py                       # Shared utilities (text normalization, language categorization)
```

## Setup

### Prerequisites

- Python 3.9+
- PyTorch
- CUDA-compatible GPU (recommended for transcription and finetuning)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
# Google Cloud Speech-to-Text
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/credentials.json'

# Deepgram
export DEEPGRAM_API_KEY='your-deepgram-api-key'

# Weights & Biases (optional, for experiment tracking)
export WANDB_API_KEY='your-wandb-api-key'

# Google Maps API (location validation)
export GOOGLE_MAPS_API_KEY='your-google-maps-api-key'

# OpenAI (street name language classification)
export OPENAI_API_KEY='your-openai-api-key'

# Mozilla Common Voice (voice cloning source audio)
export MOZILLA_API_KEY='your-mozilla-api-key'
```

## Data

Audio recordings and transcription data are not included due to privacy considerations. The pipeline expects:

- **Audio files**: `.webm` recordings in `speech_to_text/audio_files/`
- **Demographic data**: `.csv` files in `speech_to_text/demographic_data/`
- **Transcription outputs**: `.tsv` files in `speech_to_text/transcriptions/`
- **Voice cloned audio**: `.wav` files in `voice_cloning/cloned_sf_streets/` and `voice_cloning/all_us_streets/`

## License

See individual component licenses.
