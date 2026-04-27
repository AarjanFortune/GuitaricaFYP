# Tab-estimator Documentation

## What this project does

This project is a guitar tab estimation system. It takes guitar music annotation data, converts it into machine-learning input, trains a deep model, and predicts guitar tablature.

The full flow is:
- Convert GuitarSet annotation files into MIDI
- Convert MIDI into `.npz` training examples with audio features and ground truth labels
- Train a sequence model that predicts frame-level and note-level guitar tab output
- Load a trained model and evaluate tab prediction performance
- Visualize predictions and attention maps

## System flow diagrams

### High-level flow
```text
[GuitarSet JAMS + audio]   
          |
          v
   src/jams_to_midi.py
          |
          v
   [MIDI files]
          |
          v
   src/midi_to_numpy.py
          |
          v
   [data/npz/original/split/*.npz]
          |
          v
   src/train.py  (6-fold training)
          |
          v
   [model/<timestamp>/testNo0X/epoch*.model]
          |
          v
   src/predict.py
          |
          v
   [result/...] + metrics
          |
          v
   src/visualize.py
          |
          v
   [visualized plots]
```

### Function-level flow
```text
src/jams_to_midi.py
  - parses JAMS annotations
  - generates MIDI for annotated phrases

src/midi_to_numpy.py
  - loads MIDI + tempo + label data
  - extracts audio features (`cqt` or `mel_spec`)
  - writes `.npz` examples with:
      * input features
      * frame-level ground truth
      * note-level ground truth
      * tempo and lengths

src/network.py
  - defines TabEstimator model
  - defines CustomLoss and GuidedAttentionLoss
  - supports transformer/conformer encoder
  - handles optional conv stack for feature extraction

src/train.py
  - reads config.yaml
  - constructs dataset from `.npz` files
  - pads sequences with custom collate functions
  - trains on GPU with 6-fold cross-validation
  - logs losses to TensorBoard
  - saves checkpoints every 32 epochs

src/predict.py
  - loads saved model checkpoint
  - runs inference on selected test fold
  - converts raw predictions into one-hot format
  - computes precision/recall/F1 and TDR
  - saves prediction results in `result/`

src/visualize.py
  - loads saved result `.npz`
  - creates plots for prediction vs ground truth
  - saves images under `result/visualize/`
```

## Key project components

### Directories
- `data/`
  - contains numeric training examples and converted data
- `model/`
  - stores trained model checkpoints and copied config files
- `result/`
  - stores prediction results and evaluation metrics
- `tensorboard/`
  - stores TensorBoard logs from training
- `src/`
  - contains source code for preprocessing, model, training, prediction, and visualization

### Important source files
- `src/config.yaml`
  - central configuration for training and inference
- `src/jams_interpreter.py`
  - reads GuitarSet JAMS files and interprets guitar annotation data
- `src/jams_to_midi.py`
  - converts GuitarSet annotations into MIDI files
- `src/midi_to_numpy.py`
  - converts MIDI and labels into `.npz` dataset files
- `src/network.py`
  - defines the neural network model and loss functions
- `src/train.py`
  - runs training with 6-fold cross-validation and saves checkpoints
- `src/predict.py`
  - loads a saved model, performs inference, and saves evaluation results
- `src/visualize.py`
  - creates plots and visualizations from prediction results

## How the system works

### 1. Data preparation
The project assumes GuitarSet data is available and uses two conversion steps:
1. `src/jams_to_midi.py`
   - converts ground truth JAMS files into MIDI files
2. `src/midi_to_numpy.py`
   - reads MIDI and other annotation content
   - generates `.npz` files containing:
     - audio input features (`cqt` or `mel_spec`)
     - frame-level labels (`frame_tab`, `frame_F0`)
     - note-level labels (`tab`, `F0`)
     - tempo and sequence lengths

### 2. Model and training
- The model is defined in `src/network.py` as `TabEstimator`.
- It uses either a `transformer` or `conformer` encoder.
- It can optionally use a small convolutional feature stack before the encoder.
- It makes two outputs:
  - frame-level prediction
  - note-level prediction
- The custom loss combines frame loss, note loss, and optional guided attention loss.

Training is done by `src/train.py`:
- It reads `src/config.yaml` for all hyperparameters.
- It loads all `.npz` examples from `data/npz/original/split/`.
- It runs 6-fold cross-validation.
- For each fold, it trains on 5 splits and validates on 1 split.
- It saves model checkpoints at regular epoch intervals.

### 3. Inference and evaluation
`src/predict.py` does the following:
- loads a saved model checkpoint
- runs the model on test `.npz` files for a chosen fold
- converts raw outputs into one-hot tab predictions
- computes precision, recall, F1, and TDR metrics
- saves results under `result/`

### 4. Visualization
`src/visualize.py` reads prediction files and attention maps, then saves plots that help inspect how the model performed.

## Setup instructions

### Requirements
- Python 3.x
- GPU with CUDA support for training
- `requirements.txt` packages installed

### Installation
Use a Python virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Required dataset
This project depends on GuitarSet data. The code expects GuitarSet-style annotation and audio data to be available before running preprocessing.

## How to run the project

### 1. Convert annotations to MIDI
```powershell
python src/jams_to_midi.py
```

### 2. Convert MIDI to `.npz`
```powershell
python src/midi_to_numpy.py
```

### 3. Train the model
```powershell
python src/train.py
```

### 4. Predict and evaluate with a saved model
```powershell
python src/predict.py <timestamp> <epoch>
```
Example:
```powershell
python src/predict.py 202201012359 192
```

### 5. Visualize results
```powershell
python src/visualize.py <timestamp> <epoch>
```

## Configuration details

The main configuration file is `src/config.yaml`.
Below are the key settings and their meaning:

- `note_resolution`: resolution of note quantization
- `down_sampling_rate`: audio sample rate used for feature extraction
- `bins_per_octave`, `cqt_n_bins`, `hop_length`: audio feature parameters
- `train_ratio`: fraction of available non-test data used for training inside each fold
- `epoch`: number of training epochs
- `lr`: learning rate
- `seed_`: random seed
- `d_model`: model hidden dimension
- `encoder_heads`: attention heads in encoder
- `encoder_layers`: number of encoder layers
- `mode`: `tab` or `F0` mode
- `input_feature_type`: `cqt` or `melspec`
- `encoder_type`: `transformer` or `conformer`
- `use_custom_decimation_func`: controls how note downsampling is handled
- `use_conv_stack`: whether to use convolutional feature stack before encoder
- `use_galoss`: whether to apply guided attention loss

## Strict constraints for this project

These are the strict constraints and important rules for using this project correctly:

1. GPU is required for training.
   - `src/train.py` will raise an error if CUDA is not available.

2. The project expects input data in a very specific format.
   - `src/train.py` loads `.npz` files from `data/npz/original/split/*.npz`.
   - If `data/npz/original/split/` is empty or missing, training will fail.

3. The configuration file is authoritative.
   - All training and inference behavior depends on `src/config.yaml`.
   - Changing `src/config.yaml` changes the model structure and outputs.

4. Cross-validation is hard-coded to 6 folds.
   - `src/train.py` always loops over `testNo00` through `testNo05`.
   - This is not a single-fold training script.

5. Output paths are fixed.
   - Models are saved under `model/<timestamp>/testNo0X/`.
   - Results are written under `result/`.
   - TensorBoard logs are written under `tensorboard/<timestamp>/...`.

6. Mode selection is strict.
   - `mode` must be either `tab` or `F0`.
   - `input_feature_type` must be either `cqt` or `melspec`.
   - `encoder_type` must be either `transformer` or `conformer`.

7. The model architecture and loss rely on ESPnet/PyTorch dependencies.
   - If required packages are missing or incompatible, the code will fail.

8. The project is designed around GuitarSet-style tab annotation.
   - It is not a generic audio-to-tab solution for arbitrary dataset formats.

## What you should expect

- Training will likely take many hours, depending on GPU and dataset size.
- The project builds tab prediction from audio and alignment labels, not from raw audio alone.
- The output is saved as numeric predictions and evaluation metrics, not as final printable guitar tabs.

## Recommended next steps

- Verify GuitarSet data is available and converted successfully.
- Confirm `data/npz/original/split/` contains `.npz` examples.
- Run `src/train.py` on a GPU-enabled machine.
- Use the exact timestamp and epoch values from the saved `model/` folder when running prediction.

## Notes

This documentation is intentionally simple and concrete. It covers the exact project workflow, the required files, the role of each source component, and the strict constraints that must be respected when running this project.