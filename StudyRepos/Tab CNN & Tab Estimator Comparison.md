# Tab CNN & Tab Estimator Comparison

## Why compare these two projects?

Both projects try to estimate guitar tablature from audio. They use GuitarSet data, but they are built in different ways.

This file explains:
- what each project does
- how they are different
- what they are good for
- how the data and model flow works

---

## Project summaries

### `tab-cnn`

- A simpler, older implementation.
- Built with Python 2.7, Keras, and TensorFlow.
- Uses a convolutional neural network (CNN) on spectrogram patches.
- Predicts the fret and string for each audio frame.
- Good as a strong baseline and for quick experiments.

### `Tab-estimator`

- A newer, larger system.
- Built with Python 3, PyTorch, and Ignite.
- Uses a Transformer-style encoder and optional ConvStack.
- Predicts both frame-level and note-level tablature.
- Includes attention and custom loss terms.
- Better for research and deeper model experiments.

---

## Comparison table

| Feature | `tab-cnn` | `Tab-estimator` |
|---|---|---|
| Framework | Python 2.7, Keras, TensorFlow | Python 3, PyTorch, Ignite |
| Input | Preprocessed spectrogram windows | Spectrogram frames, optional ConvStack |
| Model type | 2D CNN | Transformer-style encoder + note decimation |
| Output | per-frame tab prediction | frame-level + note-level tab/F0 prediction |
| Training method | Keras fit generator | Custom Ignite engine, GPU-aware |
| Loss | string-based categorical crossentropy | frame loss + note loss + guided attention |
| Data pipeline | Spectrogram preprocessing script | JAMS → MIDI → NPZ conversion pipeline |
| Evaluation | metrics saved in CSV | tensorboard, metrics, visualization |
| Best for | baseline, fast comparison, small code | modern model, attention, research, GPU |

---

## Detailed differences

### 1. Data pipeline

`tab-cnn`:
- audio and annotations are loaded directly from GuitarSet
- a preprocessing script builds spectrogram representations
- the model reads `.npz` files with spectral data

`Tab-estimator`:
- starts from GuitarSet JAMS annotations
- converts JAMS to MIDI first
- converts MIDI to NPZ features and labels
- keeps frame-level and note-level targets

### 2. Model architecture

`tab-cnn`:
- uses a straightforward CNN
- several Conv2D layers followed by dense layers
- final output shape is `(6 strings, 21 fret classes)`
- applies a softmax for each string

`Tab-estimator`:
- optionally uses a ConvStack for input compression
- uses a Transformer encoder over time
- predicts both frame-level tab and note-level tab
- can also work in F0 mode instead of tab mode
- uses attention maps for extra analysis

### 3. Training process

`tab-cnn`:
- train/test loop is inside `model/TabCNN.py`
- 6-fold cross-validation is built in
- saves weights and predictions per fold

`Tab-estimator`:
- training is a custom loop in `src/train.py`
- uses Ignite events for iteration and epoch logging
- saves model checkpoints every 32 epochs
- logs training and validation losses to TensorBoard

### 4. Output and analysis

`tab-cnn`:
- saves prediction files and result CSV
- evaluates pitch precision/recall and tab F1

`Tab-estimator`:
- saves model checkpoints
- saves result NPZ files for predictions
- can visualize results and attention maps
- writes TensorBoard logs for loss curves

### 5. Dependencies and setup

`tab-cnn`:
- simpler requirements
- older Python and deep learning stack
- uses Keras and TensorFlow with Python 2.7

`Tab-estimator`:
- modern Python 3 environment
- PyTorch, Ignite, tensorboardX, librosa, jams, pretty_midi
- larger setup but more current tooling

---

## System diagram: `tab-cnn`

```
GuitarSet audio + annotations
         │
         ▼
Bash preprocessing script
         │
         ▼
.npy / .npz spectral files
         │
         ▼
model/TabCNN.py
   ├─ data loader
   ├─ CNN model
   └─ outputs results.csv + weights
```

## System diagram: `Tab-estimator`

```
GuitarSet JAMS + audio_mono-mic
         │
         ▼
src/jams_to_midi.py
         │
         ▼
MIDI files
         │
         ▼
src/midi_to_numpy.py
         │
         ▼
data/npz files
         │
         ▼
src/train.py
   ├─ CustomDataset
   ├─ TabEstimator model
   │    ├─ ConvStack
   │    ├─ Transformer encoder
   │    ├─ note decimation
   │    └─ frame/note heads
   ├─ CustomLoss
   └─ model checkpoints + tensorboard logs
```

---

## User flow diagram: `tab-cnn`

1. Download GuitarSet data.
2. Put it under `tab-cnn/data/GuitarSet/`.
3. Run the preprocessing script.
4. Run `python model/TabCNN.py`.
5. Review saved weights, predictions, and CSV results.

## User flow diagram: `Tab-estimator`

1. Download GuitarSet JAMS and audio.
2. Run `src/jams_to_midi.py`.
3. Run `src/midi_to_numpy.py`.
4. Check or edit `src/config.yaml`.
5. Run `src/train.py`.
6. Run `src/predict.py`.
7. Run `src/visualize.py`.

---

## Which project should you use?

- If you want a clear baseline and simpler code, choose `tab-cnn`.
- If you want a stronger modern model with attention and note-level predictions, choose `Tab-estimator`.
- If you want to learn the current best path for a real research project, `Tab-estimator` is the better option.

---

## Simple recommendation

- `tab-cnn`: fast to understand, simpler setup, good for quick tests.
- `Tab-estimator`: more powerful, slower to train, better for deeper experiments.

