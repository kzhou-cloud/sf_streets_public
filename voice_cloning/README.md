# Voice Cloning

Generate synthetic street name audio using [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) for Whisper finetuning data.

## Notebooks

- **`voice_cloning.ipynb`** -- Voice cloning for SF street names. Generates audio in `cloned_sf_streets/outputs/`.
- **`voice_cloning_us_streets.ipynb`** -- Voice cloning for US-wide street names. Generates audio in `all_us_streets/outputs/`.

Both notebooks use XTTS-v2 to clone voices from reference audio clips and synthesize "I'm on [street name]" utterances across multiple languages.

## Additional Steps to Consider

### 1. Source Speaker Audio

The voice cloning notebooks require reference audio clips from speakers of different languages. We used [Mozilla Common Voice](https://commonvoice.mozilla.org/) to obtain validated speech clips across 16 languages. You should:

- Download Common Voice datasets for your target languages via the [Mozilla Common Voice](https://commonvoice.mozilla.org/) website
- Sample a set of validated clips per language (e.g., 50 clips each)
- Place them in `random_sample_clips/<language>/` (e.g., `random_sample_clips/Spanish/`, `random_sample_clips/Korean/`)
- Each language directory should also contain a `validated.tsv` with the clip metadata

Requires a Mozilla API key: `export MOZILLA_API_KEY='your-api-key'`

### 2. Segment Extraction and Validation

The voice cloning model generates full utterances (e.g., "I'm on Cesar Chavez Boulevard"), but you will likely want to only extract the street name component. Consider:

- **Silence-based segmentation**: Use `librosa.effects.split()` to detect speech segments and extract the street name portion based on language-specific heuristics (e.g., the street name is typically the 2nd or 3rd segment after a prefix phrase)
- **Manual validation UI**: Build a Jupyter widget-based UI to listen to extractions, adjust start/end times with a slider, and save validated segments
- **Automatic extraction model**: Train a CNN on mel spectrograms of validated segments to predict start/end times for the street name within each full utterance, then snap predictions to silence boundaries for cleaner cuts

Validated segments should be saved to `<project_dir>/extracted_validated/` with the naming convention `{speaker_id}_{street_name}_{language}.wav`.
