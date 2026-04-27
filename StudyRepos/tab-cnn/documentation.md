# Data Processing Documentation

This document provides a comprehensive overview of the data processing pipeline in the TabCNN project, with an in-depth explanation of `TabDataReprGen.py`. 

The primary goal of the data processing stage is to consume raw audio and GuitarSet annotations (JAMS files) and transform them into standardized feature matrices (Spectral Representations) and categorical label arrays (Tablature Frets) that a Convolutional Neural Network can train on.

---

## 1. High-Level Data Flow

1. **Input**: Raw polyphonic guitar audio (`.wav`) and corresponding tablature annotations (`.jams`) from the GuitarSet dataset.
2. **Preprocessing**: The raw audio is downsampled, normalized, and transformed into a time-frequency spectral representation (e.g., CQT).
3. **Label Generation**: Continuous time-based MIDI pitch annotations are aligned into discrete time frames matching the audio's spectral frames. The MIDI values are mapped to fret values for each of the 6 strings.
4. **Output**: Compressed NumPy archives (`.npz`) containing the spectral feature matrices (`repr`) and one-hot encoded ground truth labels (`labels`).

---

## 2. Input and Output Formats

### **Input Format**
* **Audio (`audio/audio_mic/*.wav`)**: 
  * Source audio recorded via a microphone. Typically high-resolution (e.g., 44.1 kHz).
* **Annotations (`annotation/*.jams`)**: 
  * JSON Annotated Music Specification (JAMS) files.
  * Contains MIDI pitch curves, note events, and activations segmented by guitar string (`note_midi` array maps 6 strings).

### **Output Format (`data/spec_repr/<mode>/*.npz`)**
Each processed audio file generates a single `.npz` (NumPy Zipped) file containing a dictionary with two keys:
1. `repr` (Features): 
   * **Shape**: `(num_frames, feature_bins)`
   * **Type**: `float`
   * Represents the magnitude spectrogram. `feature_bins` size depends on the preprocessing mode (e.g., 192 bins for CQT).
2. `labels` (Ground Truth):
   * **Shape**: `(num_frames, 6, 21)`
   * **Type**: Categorical (One-hot encoded)
   * `num_frames`: Temporal frames corresponding identically to the `repr` frames.
   * `6`: The six strings of a standard guitar (E, A, D, G, B, E).
   * `21`: The 21 categorical classes (1 state for "string unplayed/closed" + 20 fret states [fret 0 through fret 19]).

---

## 3. In-Depth Breakdown of `TabDataReprGen.py`

The `TabDataReprGen` class executes the data translation. Here is a detailed breakdown of its mathematical steps and transformations.

### Step 3.1: Initialization and Constant Parameters
When `TabDataReprGen` is instantiated, it defines mapping standards:
* `string_midi_pitches = [40, 45, 50, 55, 59, 64]`: Standard guitar tuning corresponding to E2, A2, D3, G3, B3, E4.
* `highest_fret = 19`: The model limits the guitar fretboard to 20 possible notes per string (0 = open string, 1 through 19 = fretted notes).
* `num_classes = 21`: Total number of classification targets representing the guitar string state per frame (19 frets + 1 open string + 1 closed string).

### Step 3.2: Audio Preprocessing (`preprocess_audio`)
The input `.wav` data passes through a conditioning pipeline:

1. **Normalization**: The signal is normalized using `librosa.util.normalize` so its magnitude ranges strictly from -1.0 to 1.0. 
   * $y_{norm} = \frac{y}{\max(|y|)}$

2. **Downsampling**: To reduce dimensionality and remove unnecessary high-frequency content, the signal is resampled to `sr_downs = 22050` Hz.

3. **Spectral Transformation**: The mode parameter dictates the exact time-frequency representation computed. The stride between successive temporal frames is fixed at `hop_length = 512` samples.

   * **Mode `c` (Constant-Q Transform - Defaults)**:
     * Transforms a time series to the frequency domain using logarithmically spaced frequency bins (like musical notes).
     * **Formulas & Params**: 
       * `n_bins = 192` (Total frequency bins to track).
       * `bins_per_octave = 24` (Quarter-tone resolution; 2 bins per semitone).
     * Yields a magnitude representation: $| CQT(y) |$

   * **Mode `m` (Mel-Spectrogram)**:
     * Represents power spectra mapped to the Mel-scale (a perceptual scale of pitches judged by listeners to be equal in distance).
     * `n_fft = 2048` (Window size).

   * **Mode `s` (Short-time Fourier Transform - STFT)**:
     * Basic discrete Fourier transform over moving windows.
     * Magnitude calculation: $| STFT(y) |$ 

   * **Mode `cm`**: 
     * Stacks the matrices of CQT and Melspec vertically (`np.concatenate(..., axis=0)`).

*(Note: The result has axes swapped so time frames become the first axis, conforming to the network's sequential input expectations).*

### Step 3.3: Annotation & Label Construction (`load_rep_and_labels_from_raw_file`)
GuitarSet `jams` annotations are continuous in time, so they must be discretized to align with the acoustic representation.

1. **Time synchronization**: Using `librosa.frames_to_time`, the system generates timestamp milestones (in seconds) mapped dynamically strictly up to `num_frames = len(self.output["repr"])`.

2. **MIDI to Fret Conversion**:
   The script iterates over each of the 6 guitar strings. JAMS outputs the `note_midi` value overlapping each specific temporal frame.
   * **Formula**: $Fret\_Number = Round(Pitch_{MIDI}) - Open\_String_{MIDI}$
   * Note: If the frame has no active note, the array temporarily registers `-1`.

3. **Label Cleaning and Categorical Expansion** (`clean_label`, `correct_numbering`):
   In the classification space, machine learning cross-entropy requires non-negative integers. Therefore, a shift is applied:
   * **Formula**: $Class = Fret\_Number + 1$
   * By this logic:
     * Inactive string (`-1`) $\rightarrow 0$ (Closed/Unplayed)
     * Open string (`0`) $\rightarrow 1$ (Fret 0)
     * Fret 19 (`19`) $\rightarrow 20$ (Fret 19)
   * The script asserts bounds logic: `if class < 0 or class > 19` (prior to adjustment), it gets mapped to class 0 (Unplayed).

4. **One-Hot Encoding** (`categorical`):
   Finally, `to_categorical(label, num_classes=21)` expands the singular class values into probabilities.
   The resulting tensor takes the shape `(number of frames, 6 strings, 21 classes)`.

### 4. Parallel vs Sequential Execution
While `TabDataReprGen.py` generates this file by file iteratively, the script `Parallel_TabDataReprGen.py` invokes this object sequentially through python's `multiprocessing` library. This is heavily recommended as calculating 192-bin CQTs iteratively over the dataset's high-resolution audio files poses a significant computational bottleneck.

---
### Summary
For any individual frame $t$ of an audio track:
* $X_t \in \mathbb{R}^{F}$ is inputted into the CNN, where $F$ is the feature dimension sizes (e.g., $192$ for CQT).
* $Y_t \in \{0, 1\}^{6 \times 21}$ constitutes the target the CNN tries to predict (the probability of each of the 6 strings playing a respective fret state).