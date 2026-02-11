# Whisper Finetuning with "I'm on" Prefix

Finetune Whisper on synthetic voice-cloned street name audio, with "I'm on" prepended to match the recording format.

## Scripts

- **`finetuning_synthetic_data_im_on.py`** -- Finetuning on SF street names.
- **`finetuning_synthetic_data_im_on_us_streets.py`** -- Finetuning on US-wide street names.

Both scripts:
- Load an "I'm on" audio clip and prepend it to all training and test samples
- Train transcriptions include the prefix (e.g., "i'm on alemany" instead of just "alemany")
- Evaluate baseline vs finetuned models on real participant recordings

## Usage

```bash
# Local test run
python finetuning_synthetic_data_im_on.py \
    --model_size tiny \
    --language hi \
    --max_steps 50 \
    --batch_size 4 \
    --n_train_streets 5 \
    --n_test_streets 5
```

For SLURM clusters, create a submission script that calls the Python script with your desired parameters. Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SIZE` | `base` | tiny, base, small, medium, large |
| `BATCH_SIZE` | `16` | Per-GPU batch size |
| `MAX_STEPS` | `500` | Maximum training steps |
| `LEARNING_RATE` | `1e-5` | Learning rate |
| `WANDB_PROJECT` | `whisper-sf-streets-synthetic-im-on` | W&B project name |
| `SYNTHETIC_PROJECT_DIR` | `cloned_sf_streets` | Voice cloning project directory |
| `IM_ON_AUDIO` | `../im_on.webm` | Path to "I'm on" audio file |

Language indices for array jobs: 0=ar, 1=cs, 2=de, 3=es, 4=fr, 5=hi, 6=hu, 7=it, 8=ja, 9=ko, 10=nl, 11=pl, 12=pt, 13=ru, 14=tr, 15=zh-cn, 16=all

## Output

Each run creates a directory under `runs/` containing:

```
runs/<run_name>/
├── final_model/              # Finetuned model weights
├── checkpoints/              # Training checkpoints
├── comparison_results.csv    # Before/after accuracy comparison
├── config.json               # Run configuration
└── summary.json              # Run summary
```
