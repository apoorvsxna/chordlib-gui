
# Chordlib

* Chordlib is a library for automatic chord recognition. It can recognize, and help you analyse the chords that occur in an audio file.
* The heart of the project lies in the core recognition part, which you can find in the /chordlib directory, but there's also a GUI to help you use it more conveniently, with synchronized audio playback + a peek into the intermediate recognition process (such as the chromagram).

## Dependencies

chordlib-gui is built on the following key libraries:

* PyQt5: The framework for the graphical user interface.
* pyqtgraph: A plotting library used for most visualizations here.
* librosa: Used for audio loading and visualization of the chromagram.
* NumPy / SciPy: For numerical operations, and some signal processing.
* numba: For accelerating critical numerical functions through JIT compilation.

## Analysis Pipeline
In this section, I'll go over the core recognition logic. This would help you out quite a bit if contributing.
-   **1. Load Audio:** The audio file is loaded into a mono waveform.
-   **2. Calculate Onset Strength:** Spectral flux (a measure of how much the audio spectrum is changing over time) is computed to find note onsets.
-   **3. Track Beats:** The onset envelope is used to find the most probable sequence of beat frames, using dynamic programming. This is basically just guessing where a beat is and therefore, the start of a new chord. 
-   **4. Extract PCP (Chromagram):** Each frame of the audio is converted to the frequency domain using a Fourier transform (STFT). Now for each frame, we try to find the peaks (most prominent frequencies). Next, the peaks information is folded into a 12 element vector, representing the 12 notes in western music. This helps form a pitch-class profile, or a "chromagram", basically highlighting the notes observed in each frame.
-   **5. Detect Chords:** Now the PCP is segmented, using the tracked beats as boundaries. For each segment, we compute an aggregate PCP vector (median of all frames). The vector is compared to a bunch of ideal templates (of all major/minor chords), and the one with the highest correlation is declared the winner for that segment. This is done for each beat segment.
-   **6. Group Chord Events:** The raw, beat-wise chords are cleaned up (by merging consecutive identical chords), into timed events with start/end times.

##  Setup + Running the GUI

**1. Navigate to the Project Directory**

Change directory into the project's root folder.

**2. Install Dependencies**

Install the required libraries using the requirements.txt file. 
You can optionally do this in a virtual environment.

```
pip install -r requirements.txt
```

**3. Launch the Application**

Once the dependencies are installed, run the main script to start the GUI:

```
python main.py
```
**4. Run an analysis**

Once the GUI is up and running, select an audio file of your choice. The `/examples` directory has some sample files for this. Once loaded, click on the analyze button. This will run the recognition process, populate the visualizations and display the recognized chords.

##  The GUI in action
- <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/974523d7-cc23-4c90-b955-0ab0116f9fbc" />

- <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6a61028e-8f9e-4646-b60d-bb9ef51baa5d" />

- <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bbb1ee15-8fbf-45d8-8a72-7bc00bdb823f" />
