# microWakeWord Trainer

Config-driven pipeline for training custom wake word models using the
[microWakeWord](https://github.com/kahrendt/microWakeWord) framework.

microWakeWord trains a MixedNet/Inception streaming CNN that is exported as
a fully quantised TFLite model.  The final model is 50–300 KB and is designed
to run with [pymicro_wakeword](https://github.com/kahrendt/pymicro_wakeword)
(used by [linux-voice-assistant](https://github.com/OHF-Voice/linux-voice-assistant)) and
ESPHome on microcontrollers.

## Background

I built this because I wanted a custom Russian wake word — "Джарвис" (Jarvis)
— for my home voice assistant.  Russian isn't supported by any of the
off-the-shelf wake word solutions I found, and recording hundreds of your own
voice clips felt like the wrong approach.  I wanted a fully synthetic training
pipeline: pick your phrases, pick your voices, run the script, get a model.

I started with [openWakeWord](https://github.com/interkelstar/openwakeword-trainer)
and got something working, but the gap between training accuracy and real-world
recall was frustrating.  Switching to microWakeWord's streaming CNN architecture
gave much better results, and the model is a fraction of the size.

The pipeline is fully language-agnostic — the Russian example configs are just
that.  Any language with a [Piper](https://github.com/rhasspy/piper) voice
model should work the same way.

### Results on Russian "Джарвис"

Trained on 50k Piper TTS clips across 4 Russian voices (Dmitri, Denis, Irina,
Ruslan) plus ElevenLabs positives (~314k positive / ~353k negative feature
windows).  Evaluated on real audio held out of training: 50 organic
activations recorded during daily use, and 140 mined hard false-positive
clips (TV speech, clicks, knocks):

| Metric | Result |
|--------|--------|
| Recall — 50 real activations (held out of training) | **47 / 50 (94%)** |
| False positives — 140 mined hard cases | **1 / 140** |
| Previous model on the same test sets | 50/50 recall, but 91/140 FP |
| Production cutoff in daily use | **0.9** |
| Model size | **82.5 KB** (streaming, uint8 quantised) |
| Val accuracy | **99.77%** |

The model runs continuously in production.  The biggest win came from
FP-mining: collecting 208 real false-positive clips from daily use and
re-extracting negative features with dense windowing (1 window per 10 ms)
dropped hard-case false positives from 91/140 to 1/140.  Lowering the cutoff
from 0.97 to 0.9 then recovered 5 more real activations at no measurable FP
cost — the 3 remaining misses sit far below the threshold (~0.2–0.5) and are
not recoverable by tuning alone.

---

## Requirements

- Linux on x86_64 or aarch64 (tested on Ubuntu 22.04, Debian 12, WSL2; the
  aarch64 path targets the NVIDIA DGX Spark — see below)
- [uv](https://docs.astral.sh/uv/) — it fetches Python 3.12 and every
  dependency itself; nothing is installed system-wide
- ~15 GB disk space for training data
- Internet access for HuggingFace and GitHub downloads
- `ffmpeg` only for the optional `elevenlabs` / `ha-tts` clip generators

## Quick Start

```bash
# Install uv once, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone (with the pinned microWakeWord submodule) and set up
git clone --recurse-submodules https://github.com/interkelstar/microwakeword-trainer.git
cd microwakeword-trainer
uv sync          # creates .venv/ from uv.lock
uv run mww setup # downloads the Piper binary for this CPU, verifies the install

# Copy the example config and customise it for your wake word
cp example_ru_jarvis.yaml my_wakeword.yaml
# Edit my_wakeword.yaml: set model_name, target_phrases, voices, negatives

# Run the full training pipeline
uv run mww run --config my_wakeword.yaml
```

Already cloned without `--recurse-submodules`? Run
`git submodule update --init` before `uv sync`.

Everything is driven by the `mww` command; `uv run mww --help` lists the
subcommands and `uv run mww <command> --help` shows each one's options:

| Command             | What it does                                                   |
|---------------------|----------------------------------------------------------------|
| `mww setup`         | Download the Piper binary for this CPU, verify the environment |
| `mww run`           | Run training pipeline phases from a YAML config                |
| `mww test`          | Run a TFLite model over WAV files, pymicro_wakeword-style      |
| `mww batch-test`    | Summarise which false-positive clips still trigger             |
| `mww ab-test`       | Compare two models on the same false-positive clips            |
| `mww elevenlabs`    | Generate extra positives with the ElevenLabs API               |
| `mww ha-tts`        | Generate clips from Home Assistant's TTS engines               |
| `mww channel-probe` | Put clips through the room/noise channel for inspection        |

If you would rather not prefix every command with `uv run`, activate the
environment once with `source .venv/bin/activate` and call `mww` directly.

Output files created in the working directory after training:

- `<model_name>_mww.tflite` — TFLite model ready for pymicro_wakeword
- `<model_name>_mww.json` — Deployment manifest with inference parameters

## Training Phases

The pipeline runs these phases in order:

| Phase      | What it does                                                        | Time (CPU)  |
|------------|---------------------------------------------------------------------|-------------|
| `setup`    | Download Piper for this CPU; verify microwakeword, TF, etc.         | seconds     |
| `voices`   | Download Piper ONNX voice models from HuggingFace                   | 1–5 min     |
| `generate` | Synthesise positive + adversarial negative TTS clips via piper      | 6–10 hours  |
| `features` | Extract pymicro_features spectrograms + download HF negatives       | 1–3 hours   |
| `train`    | Build and train the MixedNet model with SpecAugment                 | 0.5–3 hours |
| `export`   | Convert to streaming TFLite (uint8 quantised) + write JSON manifest | 5–15 min    |
| `test`     | Evaluate TFLite model on test clips (optional, not run in "all")    | minutes     |

Run individual phases with `--phase <name>`, or several at once as a
comma-separated list (they always run in pipeline order):

```bash
uv run mww run --config my_wakeword.yaml --phase train
uv run mww run --config my_wakeword.yaml --phase features,train,export
uv run mww run --config my_wakeword.yaml --phase test
```

Every phase skips work whose output already exists, so an interrupted run can
simply be started again.

## Config Reference

See `example_ru_jarvis.yaml` for a fully annotated example. Key sections:

```yaml
model_name: ru_jarvis        # Output filename prefix (ASCII, no spaces)

target_phrases:              # What the model should fire on
  - Джарвис
  - Джарвис!
  - Джарвис?

training:
  n_samples: 50000           # TTS clips for training
  n_samples_val: 10000       # TTS clips for validation

voices:
  primary:                   # For positive clips + same-language negatives
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU
    models:
      ru_RU-dmitri-medium: dmitri/medium
  secondary:                 # For cross-language negative phrases (optional)
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main
    models:
      en_US-amy-medium: en/en_US/amy/medium

negative_phrases:            # Acoustically similar words to reject
  - Джар
  - Алиса
  - Jarring                  # Latin script → synthesised with secondary voices

microwakeword:
  architecture: mixednet     # mixednet (default) or inception
  spectrogram_length: 204    # input frames per inference window
  stride: 3                  # frames consumed per streaming step (3 × 10ms = 30ms)
  training_steps: 10000      # gradient steps
  batch_size: 128
  learning_rate: 0.001
  positive_class_weight: 1.5 # raise if recall is too low
  negative_class_weight: 1   # raise if false positive rate is too high
  probability_cutoff: 0.9    # detection threshold (used at inference time)
  sliding_window_size: 10    # averaging window size (used at inference time)
```

### Understanding spectrogram_length and stride

microWakeWord uses pymicro_features to extract 40-bin mel spectrogram frames
at 10 ms intervals.  The model processes `stride` frames per inference call
and maintains a ring buffer of internal state.

The `spectrogram_length` determines how many frames the model can look back
when making each decision.  Larger values give more context but increase model
size and training memory.

`probability_cutoff` and `sliding_window_size` are used only at inference
time — re-run `--phase export` after changing them.

## Deployment

Copy both output files to your linux-voice-assistant config directory:

```bash
cp ru_jarvis_mww.tflite /path/to/linux-voice-assistant/models/
cp ru_jarvis_mww.json   /path/to/linux-voice-assistant/models/
```

The JSON manifest format:

```json
{
  "type": "micro",
  "wake_word": "Джарвис",
  "model": "ru_jarvis_mww.tflite",
  "micro": {
    "probability_cutoff": 0.9,
    "sliding_window_size": 10,
    "feature_step_size": 10
  }
}
```

The `type: micro` field is required — the linux-voice-assistant loader uses it
to select the pymicro_wakeword inference backend.  The exporter also writes
`author`, `trained_languages`, `version`, `micro.tensor_arena_size`, and
`micro.minimum_esphome_version` for ESPHome compatibility.

To change the detection threshold without retraining, edit `probability_cutoff`
in the JSON file directly, or change it in your YAML and re-run `--phase export`.

## Iterative Improvement — Reducing False Positives

The most effective way to reduce false positives is to collect negative examples
from your real environment and retrain.

**Workflow:**

1. Run the assistant for a day and note when it triggers incorrectly.
2. Record the audio that caused false positives as a WAV file (16 kHz, mono).
3. Copy the WAV to `training/output/<model_name>/user_negatives/`.
4. Re-run feature extraction and training:

```bash
rm training/output/<model_name>/mww_features/negative_*.npy
uv run mww run --config my_wakeword.yaml --phase features,train,export
```

The `user_negatives/` directory is processed with dense windowing (1 window
per 10 ms) to maximise coverage of the triggering audio.

**Important:** Do NOT put actual wake word recordings in `user_negatives/`.
Only put audio that triggered incorrectly (TV speech, background noise, etc.).

## Testing a Model

Use `mww test` to evaluate a trained model against WAV files.  It
simulates the exact pymicro_wakeword inference loop including the sliding
window averaging:

```bash
# Test against all WAV files in record/
uv run mww test --model my_model_mww.tflite

# Test specific files
uv run mww test --model my_model_mww.tflite record/sample1.wav record/sample2.wav

# Output shows per-frame probabilities and a TRIGGERED / no trigger result
```

The JSON manifest is found automatically next to the model; pass
`--config` to use a different one.

Two companion tools help with false-positive mining:

```bash
# Batch-test a directory of FP clips at the production cutoff
uv run mww batch-test --model my_model_mww.tflite fp_clips/

# A/B compare two model versions on the same FP clips
uv run mww ab-test old_mww.tflite new_mww.tflite fp_clips/
```

## ElevenLabs High-Quality Positives (Optional)

`mww elevenlabs` generates additional high-quality TTS clips using the
ElevenLabs API (requires an API key with credits, and `ffmpeg`).  These are
automatically picked up during feature extraction and can improve the
naturalness of the positive class:

```bash
ELEVENLABS_API_KEY=YOUR_KEY uv run mww elevenlabs --config my_wakeword.yaml
```

Clips are saved to `training/output/<model_name>/elevenlabs_positive/`.

## Disk Space

After a full training run, the `training/` directory contains:

```
training/piper_binary/           # piper executable + shared libs for this CPU (~150 MB)
training/piper_models/           # ONNX voice models (~280 MB for 4 voices)
training/output/<model_name>/
  positive_train/                # TTS positive clips (~4 GB for 50k)
  positive_test/                 # TTS validation clips (~800 MB for 10k)
  negative_train/                # Adversarial negative TTS clips
  negative_test/
  elevenlabs_positive/           # ElevenLabs clips (optional)
  user_negatives/                # Real environment negatives (optional)
  mww_features/                  # Extracted spectrograms + HF negatives
  model/                         # Saved Keras model weights
```

To clean up after training:

```bash
rm -rf .venv training/
```

## Running on an NVIDIA DGX Spark

The Spark is an aarch64 machine (Grace CPU, 20 cores, 128 GB unified memory)
with a Blackwell GB10 GPU.  The same commands work there unchanged and
directly on the host — no container:

```bash
git clone --recurse-submodules https://github.com/interkelstar/microwakeword-trainer.git
cd microwakeword-trainer
uv sync
uv run mww setup     # fetches piper_linux_aarch64; reports "Training device: CPU"
uv run mww run --config my_wakeword.yaml
```

What is different on aarch64:

- **Piper:** `mww setup` picks the aarch64 build of the standalone Piper
  binary automatically.
- **Dependencies:** `uv.lock` is resolved for both x86_64 and aarch64 Linux,
  and every package in it ships an aarch64 wheel, so no compiler is needed.
  (This is why `audiomentations` is held below 0.39: later versions hard-depend
  on `python-stretch`, which has no aarch64 wheel.)
- **Training runs on the CPU.**  PyPI's TensorFlow for Linux aarch64 is a
  CPU-only build and there is no TensorFlow GPU wheel for the GB10 (sm_121),
  so the GPU is not used.  The `cuda` extra is a no-op on aarch64.

### What CPU-only costs

Only the `train` phase could use a GPU at all — everything else is CPU-bound
on any machine:

| Phase      | Uses GPU elsewhere? | On the Spark                                                 |
|------------|---------------------|--------------------------------------------------------------|
| `generate` | no (Piper is CPU)   | Same as today: ~16 parallel Piper workers on 20 cores        |
| `features` | no                  | Same; 128 GB unified memory comfortably fits the ~40 GB peak |
| `train`    | yes                 | Roughly 0.5–3 h on CPU vs ~15–45 min on a discrete GPU       |
| `export`   | no                  | Minutes                                                      |

The model is tiny (~80 KB) and the training input pipeline is a Python
generator, so a GPU would speed `train` up by roughly 2–4×, not by an order
of magnitude.  Since TTS generation and feature extraction take most of the
8–14 hours, a full run on the Spark should take about as long as on an x86
workstation with a GPU — at most an hour or two longer.

On an x86_64 machine with an NVIDIA GPU, `uv sync --extra cuda` installs the
CUDA runtime from PyPI so TensorFlow can train on the GPU.  `mww run --cpu`
forces CPU training anywhere.

## How It Works

microWakeWord uses the pymicro_features frontend (matching the on-device
inference pipeline on microcontrollers):

1. Audio is chunked into 10 ms frames of 160 samples at 16 kHz.
2. Each chunk is processed by `MicroFrontend` from the `pymicro-features`
   package, producing a 40-bin log-mel spectrogram frame.
3. The streaming model processes `stride` frames per call, maintaining internal
   convolutional state across calls.
4. A sliding window of `sliding_window_size` consecutive probabilities is
   averaged; when the mean exceeds `probability_cutoff`, the wake word fires.

**Feature scale:** pymicro_features outputs values in [0, ~26] (float32).
This range must match at training time and inference time.  The HuggingFace
negative dataset (`kahrendt/microwakeword`) uses a uint16 scale and is
automatically rescaled to match during feature extraction.

## License

MIT
