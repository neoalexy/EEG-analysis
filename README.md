# EEG Signal Analysis and Visualization - MNE-Python
# Table of Contents

1. [Part One – Simulated EEG Data and Visualization](#part-one--simulated-eeg-data-and-visualization)  
   - [Sensor Display (2D and 3D)](#sensor-display-2d-and-3d)  
   - [Raw EEG Signal and Average Reference](#raw-eeg-signal-and-average-reference)  
   - [Segmentation (Epochs) and ERP Analysis](#segmentation-epochs-and-erp-analysis)  
   - [Topomap and PSD Visualizations](#topomap-and-psd-visualizations)  
   - [Heatmap and ERP Signal Comparison for Different Stimuli](#heatmap-and-erp-signal-comparison-for-different-stimuli)  

2. [Part One – CSP Analysis and Classification (Motor Imagery)](#part-one--csp-analysis-and-classification-motor-imagery)  
   - [Data Preparation and Filtering](#data-preparation-and-filtering)  
   - [Defining CSP and Classifier Parameter Variants](#defining-csp-and-classifier-parameter-variants)  
   - [CSP Computation, Training, and Cross-Validation](#csp-computation-training-and-cross-validation)  
   - [Results Visualization and Variant Comparison](#results-visualization-and-variant-comparison)  

3. [Part Two – Real EEG Data and Classification](#part-two--real-eeg-data-and-classification)  
   - [Loading EDF and TSV Files](#loading-edf-and-tsv-files)  
   - [Signal Segmentation into Epochs](#signal-segmentation-into-epochs)  
   - [Applying Previous CSP Analysis and Classification](#applying-previous-csp-analysis-and-classification)  
   - [Results and Classification Accuracy Visualization](#results-and-classification-accuracy-visualization)


## Part One - First Example

This analysis demonstrates processing and visualization of EEG signals using the MNE-Python library with the goal of demonstrating basic steps in EEG signal processing including filtering, segmentation (epochs), averaging (ERP). For better understanding of parameter changes in EEG signal analysis, comparisons of similar waves with some types of modifications are presented.

---

## 1. Sensor Visualization - Topomap 
![](/pics/1.png)

Primarily showing the position of EEG sensors on the head surface in 2D format. This display is useful for orientation of where electrodes are placed, which is crucial for interpreting signals from different brain locations.
```python 
raw.plot_sensors(kind='topomap', show_names=True)
```
---

## 2. Sensor Visualization - 3D Display
![](/pics/2.png)

This image shows sensors in three-dimensional space allowing us better spatial representation of electrode positions relative to the head shape itself and also helps in understanding the measurement geometry.
```python 
raw.plot_sensors(kind='3d')
```
---

## 3. Initial EEG Signal Without Average Reference 
![](/pics/3.png)

Raw EEG signal is shown on the first 5 channels without any corrections or changes. This is the basic display where noise and other variations can be observed.
```python 
raw.plot(n_channels=5, proj=False)
```
---

## 4. EEG Signal With Applied Average EEG Reference 
![](/pics/4.png)

After applying average reference the signal is cleaner and more stable. Average reference means subtracting the average signal value from all channels from each individual channel, resulting in a more stable and reliable EEG recording.
```python 
raw.set_eeg_reference('average')
raw.plot(n_channels=5, proj=True)
```

---

## 5. Epoch Visualization 
![](/pics/5.png)

Signal is segmented into epochs which represent time intervals around stimuli. This allows analysis of brain responses that are temporally linked to specific events. The image shows how the segmented EEG signal looks.
```python 
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7, preload=True)
epochs.drop_bad(reject=dict(eeg=100e-6, eog=21e-6))
epochs.plot()
```
---

## 6. ERP Signal for Channel eeg01 - Auditory Stimulus 
![](/pics/6.png)

ERP (Event-Related Potential) represents the average brain response to a specific stimulus. This image shows the average response to auditory stimulus ("auditory/left") on channel eeg01. Spatial colors help highlight active parts.
```python 
erp1 = epochs['auditory/left'].average()
erp1.plot(picks="eeg01", spatial_colors=True)
```

---

## 7. ERP Signal for Channel eeg01 - Visual Stimulus 
![](/pics/7.png)

Comparison with previous display — ERP response to visual stimulus ("visual/left") on the same channel, this time without spatial color scheme which can be useful for different signal interpretation.
```python 
erp2 = epochs['visual/left'].average()
erp2.plot(picks="eeg01", spatial_colors=False)
```

---

## 8. Topomap Activity Distribution 
![](/pics/8.png)

Shows spatial distribution of brain activity at specific time moments after stimulus. This visualization provides insight into localization of active brain regions during different phases of brain response.
```python 
fig1 = erp1.plot_topomap(times=[0.1, 0.2, 0.3])
fig1.suptitle("Topomap - Initial Version (auditory/left)")
```
---

## 9. Topomap Activity Distribution - Modified Version 
![](/pics/9.png)

Similar to previous but with different parameters (without contours). Allows better understanding of changes in spatial activity.
```python 
fig2 = erp2.plot_topomap(times=[0.1, 0.2, 0.3], contours=0)
fig2.suptitle("Topomap - Second Version (visual/left)")
```

---

## 10. Joint ERP Signal and Topomap Display 
![](/pics/10.png)     ![](/pics/11.png)

Top image - combined display showing how ERP waveform and spatial brain activity change over time. Bottom image - similar concept as previous image but with modified time intervals and parameters for better clarity and different insights into data.
```python 
erp1.plot_joint(title="ERP + Topomap - Initial Version (auditory/left)", times=[0.1, 0.2])
erp2.plot_joint(title="ERP + Topomap - Second Version (visual/left)", times=[0.2, 0.3])
```

---


## 12. Power Spectral Density (PSD)
![](/pics/12.png)

Display of signal power distribution by frequencies for channel eeg01. This facilitates identification of dominant frequencies i.e. alpha, beta, gamma waves, and can also indicate potential artifacts or specific brain activities.
```python 
psd1 = epochs['auditory/left'].compute_psd(fmin=1, fmax=40)
psd1.plot(picks="eeg01", average=True, spatial_colors=True)
```
---

## 13. PSD Analysis for Channel eeg02
![](/pics/13.png)

Similar analysis as previous but for second channel (eeg02) and with slightly different parameters (without averaging).
```python 
psd1.plot(picks="eeg02", average=False, spatial_colors=False)
```
---

## 14. Heatmap of eeg01 Signal
![](/pics/14.png)

Heatmap shows signal power through time and frequency where sigma parameter determines display smoothness. Smaller sigma gives more detailed and sharper display but with more noise.
```python 
epochs['auditory/left'].plot_image(picks="eeg01", sigma=0.1, title="Heatmap eeg01 - Initial Version")
```
---

## 15. Heatmap of eeg01 Signal with Larger Sigma 
![](/pics/15.png)

By increasing sigma parameter we get "smoother" signal display, which facilitates observation of larger patterns and reduces noise but may omit fine details.
```python 
epochs['auditory/left'].plot_image(picks="eeg01", sigma=0.3, title="Heatmap eeg01 - Second Version")
```
---

## 16. Comparison of ERP Signals for Auditory and Visual Stimuli 
![](/pics/16.png)

This image shows comparison of ERP signals for two different types of stimuli on channel eeg01. Visible differences in amplitudes and temporal patterns indicate different brain processes activated by auditory and visual stimuli.
```python 
plt.plot(erp1.times, erp1.data[erp1.ch_names.index("eeg01")] * 1e6, label='Auditory Left')
plt.plot(erp2.times, erp2.data[erp2.ch_names.index("eeg01")] * 1e6, label='Visual Left', linestyle='--')
```
---
## 17. Complete Code
```python 
import mne
import matplotlib.pyplot as plt
from pathlib import Path
#Part One-First Example - erps_analysis.py
# loading sample EEG data from MNE sample dataset
root = mne.datasets.sample.data_path() / "MEG" / "sample"
raw_file = root / "sample_audvis_filt-0-40_raw.fif"
raw = mne.io.read_raw_fif(raw_file, preload=True) #entire file loaded immediately into RAM - faster processing

# loading events
events_file = root / "sample_audvis_filt-0-40_raw-eve.fif"
events = mne.read_events(events_file)

#signal is cropped (for faster processing) 
raw.crop(tmax=90)
#only events that fall within this interval are filtered
events = events[events[:, 0] <= raw.last_samp]

#selecting only eeg and eog channels 
raw.pick(["eeg", "eog"])
#EEG 01 -> eeg01
raw.rename_channels({ch: ch.replace(" 0", "").lower() for ch in raw.ch_names})

#2d and 3d sensor display
raw.plot_sensors(kind='topomap', show_names=True)
raw.plot_sensors(kind='3d')

# initial signal version without ref - more noise
raw.plot(n_channels=5, proj=False, title="Initial Version - Without Average Reference")
# average reference added - average value of all channels subtracted from each individual channel - better quality
raw.set_eeg_reference('average')
raw.plot(n_channels=5, proj=True, title="Second Version - With Average Reference")

#high-pass filter for signal filtering
#removes slow changes and drifts
raw.filter(l_freq=0.1, h_freq=None)

#mapping events of interest
event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "face": 5,
    "buttonpress": 32,
}

# creating epochs
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7, preload=True)

#bad epochs that could corrupt ERP are rejected 
#if amplitude is too high - noise, movement, blinking
reject_criteria = dict(eeg=100e-6, eog=21e-6)
epochs.drop_bad(reject=reject_criteria)

#epoch display
epochs.plot()

#erp analysis - average signal of all epochs of same type
erp1 = epochs['auditory/left'].average()
erp2 = epochs['visual/left'].average()

#display of average brain response to aud. and vis. stimulus
#changing spatial_colors parameter - colors show channel by type
erp1.plot(picks="eeg01", spatial_colors=True, titles='ERP eeg01 - Initial Version (auditory/left)')
erp2.plot(picks="eeg01", spatial_colors=False, titles='ERP eeg01 - Second Version (visual/left)')

#topomap display - where on head most activity occurs after stimulus
fig1 = erp1.plot_topomap(times=[0.1, 0.2, 0.3])
fig1.suptitle("Topomap - Initial Version (auditory/left)")

fig2 = erp2.plot_topomap(times=[0.1, 0.2, 0.3], contours=0)
fig2.suptitle("Topomap - Second Version (visual/left)")

#combined ERP waveform and topomap display
erp1.plot_joint(title="ERP + Topomap - Initial Version (auditory/left)", times=[0.1, 0.2])
erp2.plot_joint(title="ERP + Topomap - Second Version (visual/left)", times=[0.2, 0.3])

#PSD analysis - power by frequencies - dominant rhythms(alpha,beta,gamma)
psd1 = epochs['auditory/left'].compute_psd(fmin=1, fmax=40)
fig1 = psd1.plot(picks="eeg01", average=True, spatial_colors=True)
fig1.suptitle("PSD - Initial Version eeg01")

fig2 = psd1.plot(picks="eeg02", average=False, spatial_colors=False)
fig2.suptitle("PSD - Second Version eeg02")


#epoch display through heatmap for comparing epoch similarities
#changes in sigma parameter, smaller - more details
epochs['auditory/left'].plot_image(picks="eeg01", sigma=0.1, title="Heatmap eeg01 - Initial Version")
epochs['auditory/left'].plot_image(picks="eeg01", sigma=0.3, title="Heatmap eeg01 - Second Version (larger sigma)")

#manual comparison of auditory and visual erp signal on eeg01(time and amp) 
plt.figure()
plt.plot(erp1.times, erp1.data[erp1.ch_names.index("eeg01")] * 1e6, label='Auditory Left')
plt.plot(erp2.times, erp2.data[erp2.ch_names.index("eeg01")] * 1e6, label='Visual Left', linestyle='--')
plt.title("ERP Comparison - eeg01: Initial vs Second Version (stimulus)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [μV]")
plt.legend()
plt.show()
```
# Conclusion

This analysis demonstrates basic techniques in EEG signal processing with emphasis on visualization and data comparison before and after different processing steps. Multiple types of visualizations were applied (topomap, ERP, PSD, heatmap).



## Part One - Second Example - Motor Imagery: CSP Analysis and Classification

In this part, the **Common Spatial Patterns (CSP)** method is used for extracting spatial-frequency patterns from EEG signals related to motor imagery tasks. After feature extraction, classification is used to distinguish mental states.

### 1. Data Loading and Preparation

Data for subject 1 is loaded for two sessions containing motor imagery tasks (runs 6 and 10). Then data is filtered in frequency range from 7 to 30 Hz to extract relevant oscillations, and segmented into epochs.
```python
raw_files = [eegbci.load_data(subject, run) for run in runs]
raws = [read_raw_edf(f[0], preload=True, verbose=False) for f in raw_files]
raw = concatenate_raws(raws)

raw.pick_types(eeg=True, exclude='bads')  # EEG channel selection
raw.set_eeg_reference('average', projection=True)  # Average EEG reference
raw.filter(7., 30., fir_design='firwin', verbose=False)  # Band-pass filter

events, event_id = mne.events_from_annotations(raw)  # Events and their ID map
epochs = mne.Epochs(raw, events, event_id=wanted_events, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)  # Signal segmentation into epochs
```
Filter retains oscillations in beta and alpha range (important for motor imagery).
Epochs represent time intervals around stimuli that we use for analysis.

### 2. Defining Experiment Parameters

Experiment is conducted in four variants where parameters are changed:

- Number of CSP components (4 or 6)
- Classifier (LDA or SVM with linear kernel)
- Frequency range (7-30 Hz or 8-26 Hz)
```python
params_variants = [
    {'n_csp': 4, 'clf': LDA(), 'freq': (7, 30), 'desc': 'Original Version'},
    {'n_csp': 6, 'clf': LDA(), 'freq': (7, 30), 'desc': 'Variant 1'},
    {'n_csp': 4, 'clf': SVC(kernel="linear"), 'freq': (7, 30), 'desc': 'Variant 2'},
    {'n_csp': 4, 'clf': LDA(), 'freq': (8, 26), 'desc': 'Variant 3'},
]
```
These variants allow assessment of parameter impact on classification performance.
By changing number of CSP components we change how many spatial-discriminative features we extract. Different classifiers and frequency ranges can affect accuracy.

### 3. Running CSP Analysis and Classification

For each variant:

- Data is filtered by given frequency range
- CSP components are calculated that emphasize differences between classes
- Classifier is trained and evaluated using 5-fold cross-validation
- Average accuracy and standard deviation are calculated

```python
epochs_filtered = epochs.copy().filter(freq[0], freq[1], fir_design='firwin', verbose=False)
X = epochs_filtered.get_data()  # Data in shape (epochs, channels, time)
y = labels

csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
clf_pipe = Pipeline([('CSP', csp), ('Classifier', clf)])

scores = cross_val_score(clf_pipe, X, y, cv=5, n_jobs=1)
mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"Accuracy: {mean_score:.3f} ± {std_score:.3f}")
```
CSP projection emphasizes differences between classes.
Pipeline combines CSP and classifier into one unit.
Cross-validation gives model generalization estimate.

### 4. Results and Comparison

| Variant      | Accuracy (mean ± std) |
|--------------|------------------------|
| Original     |     0.842 ± 0.112     |
| Variant 1    |     0.794 ± 0.085     |
| Variant 2    |     0.728 ± 0.087     |
| Variant 3    |     0.864 ± 0.043     |

Best result achieved with Variant 3 with 4 CSP components, LDA classifier and frequency range 8-26 Hz.

### 5. Visualization Comparison

![Comparison of accuracy of different CSP analysis variants](/pics/grafikon_bar.png)

Chart clearly shows which variant gave best results and accuracy variability through standard deviation.
### 6. Complete Code
```python
#Part One-Second Example - csp_analysis.py
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# loading edf files; 6 and 10 contain motor imagery, combining into one raw
def load_data(subject=1, runs=[6, 10]):
    raw_files = [eegbci.load_data(subject, run) for run in runs]
    raws = [read_raw_edf(f[0], preload=True, verbose=False) for f in raw_files]
    raw = concatenate_raws(raws)

#extracting only eeg channels
    raw.pick_types(eeg=True, exclude='bads')
    #average reference for signal stabilization
    raw.set_eeg_reference('average', projection=True)
#band-pass filter (alpha and beta waves)
    raw.filter(7., 30., fir_design='firwin', verbose=False)

#extracting only needed events from annotations
    events, event_id = mne.events_from_annotations(raw)
    print("Event ID map:", event_id)

    #T0-rest and T1-imagery
    wanted_events = {k: event_id[k] for k in event_id if k in ['T0', 'T1']}
    if not wanted_events:
        raise RuntimeError("No needed events T0 or T1")
#signal segmentation into epochs
    epochs = mne.Epochs(raw, events, event_id=wanted_events, tmin=0.5, tmax=2.5,
                        baseline=None, preload=True, verbose=False)
#labels important for supervised learning
    labels = epochs.events[:, -1]
    return epochs, labels


#Definition of experiment variants
params_variants = [
    {'n_csp': 4, 'clf': LDA(), 'freq': (7, 30),
     'desc': 'Original Version: 4 CSP components, LDA classifier, frequency range 7-30 Hz'},

    {'n_csp': 6, 'clf': LDA(), 'freq': (7, 30),
     'desc': 'Variant 1: 6 CSP components, LDA classifier, frequency range 7-30 Hz'},

    {'n_csp': 4, 'clf': SVC(kernel="linear"), 'freq': (7, 30),
     'desc': 'Variant 2: 4 CSP components, SVC (linear kernel), frequency range 7-30 Hz'},

    {'n_csp': 4, 'clf': LDA(), 'freq': (8, 26),
     'desc': 'Variant 3: 4 CSP components, LDA classifier, frequency range 8-26 Hz'},
]


#Function for running analysis for one variant
def run_variant(epochs, labels, n_csp, clf, freq, description):
    
    print(f"\nStarting analysis: {description}")

    epochs_filtered = epochs.copy().filter(freq[0], freq[1], fir_design='firwin', verbose=False)
  
    # Shape: (epochs, channels, time)
    X = epochs_filtered.get_data()  
 
    y = labels

    # Creating CSP object
    csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)

    # Creating pipeline: CSP + classifier
    clf_pipe = Pipeline([('CSP', csp), ('Classifier', clf)])

    # Cross-validation (5-fold)
    scores = cross_val_score(clf_pipe, X, y, cv=5, n_jobs=1)
#average accuracy and std deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Accuracy: {mean_score:.3f} ± {std_score:.3f}")

    return mean_score, std_score


#Main part
if __name__ == "__main__":
    epochs, labels = load_data()

    results = []
    for p in params_variants:
        mean_acc, std_acc = run_variant(epochs, labels, p['n_csp'], p['clf'], p['freq'], p['desc'])
        results.append({'desc': p['desc'], 'mean_acc': mean_acc, 'std_acc': std_acc})

    print("\n--- Summary of all variants ---")
    for r in results:
        print(f"{r['desc']} -> Accuracy: {r['mean_acc']:.3f} ± {r['std_acc']:.3f}")

    #Charts-comparison
    

    labels = [r['desc'] for r in results]
    accuracies = [r['mean_acc'] for r in results]
    std_devs = [r['std_acc'] for r in results]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(x, accuracies, yerr=std_devs, capsize=8, color='cornflowerblue')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{acc:.2f}",
                ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Comparison of Accuracy of Different CSP Analysis Variants')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')

    plt.tight_layout()
    plt.show()
    
#wrapper function for second part
def run(raw):
    import mne
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import matplotlib.pyplot as plt

    #creating events from annotations
    events, event_id = mne.events_from_annotations(raw)
    print("Event ID map:", event_id)

    #displaying available keys
    if len(event_id) < 2:
        raise RuntimeError("Not enough different events.")

    #2 most frequent
    sorted_keys = sorted(list(event_id.keys()))
    chosen_events = {sorted_keys[0]: event_id[sorted_keys[0]],
                     sorted_keys[1]: event_id[sorted_keys[1]]}
    print("Used events:", chosen_events)

    #creating epochs
    epochs = mne.Epochs(raw, events, event_id=chosen_events,
                        tmin=0.5, tmax=2.5, baseline=None,
                        preload=True, verbose=False)

    labels = epochs.events[:, -1]

    params_variants = [
        {'n_csp': 4, 'clf': LDA(), 'freq': (7, 30),
         'desc': 'Original Version: 4 CSP components, LDA classifier, 7-30 Hz'},
        {'n_csp': 6, 'clf': LDA(), 'freq': (7, 30),
         'desc': 'Variant 1: 6 CSP, LDA, 7-30 Hz'},
        {'n_csp': 4, 'clf': SVC(kernel="linear"), 'freq': (7, 30),
         'desc': 'Variant 2: 4 CSP, SVC (linear), 7-30 Hz'},
        {'n_csp': 4, 'clf': LDA(), 'freq': (8, 26),
         'desc': 'Variant 3: 4 CSP, LDA, 8-26 Hz'},
    ]

    results = []
    for p in params_variants:
        print(f"\nRunning: {p['desc']}")
        epochs_filtered = epochs.copy().filter(p['freq'][0], p['freq'][1], fir_design='firwin', verbose=False)
        X = epochs_filtered.get_data()
        y = labels

        csp = CSP(n_components=p['n_csp'], reg=None, log=True, norm_trace=False)
        clf_pipe = Pipeline([('CSP', csp), ('Classifier', p['clf'])])

        scores = cross_val_score(clf_pipe, X, y, cv=5, n_jobs=1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Accuracy: {mean_score:.3f} ± {std_score:.3f}")
        results.append({'desc': p['desc'], 'mean_acc': mean_score, 'std_acc': std_score})

    # Visualization
    labels_bar = [r['desc'] for r in results]
    accuracies = [r['mean_acc'] for r in results]
    std_devs = [r['std_acc'] for r in results]

    x = np.arange(len(labels_bar))
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(x, accuracies, yerr=std_devs, capsize=8, color='cornflowerblue')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{acc:.2f}",
                ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Comparison of CSP Variant Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=25, ha='right')
    plt.tight_layout()
    plt.show()



```

### Conclusion

This analysis shows how choice of CSP method parameters and classifier affects precision of decoding motor imagery from EEG signals. 
CSP efficiently emphasizes discriminative spatial patterns while LDA classifier gives better performance in this case compared to SVM with linear kernel. 
Additionally, signal frequency range is significant factor for result optimization.

---
# Part Two - Real Data: Loading, Segmentation and CSP Analysis with Classification

In this part, we work with loading real EEG data from EDF files and events from TSV files. The data is filtered, segmented into epochs based on given events, and then the existing CSP analysis (PartOne_1) and classification for distinguishing motor tasks is applied.

### 1. Data Loading and Preparation

First, the EEG signal is loaded from the EDF file and then average reference and filtering in the frequency range from 7 to 30 Hz is applied to extract significant oscillations.

```python
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
raw.set_eeg_reference('average', projection=True)
raw.filter(7., 30., fir_design='firwin', verbose=False)
```

Then the TSV file with events is loaded and events relevant to the task are filtered (e.g., labels T0, T1, T2), and an event_id map is defined that assigns a unique number to each class.

```python
events_df = pd.read_csv(events_tsv_path, sep='\t')
wanted = [val for val in sorted(set(events_df['value'])) if any(t in val for t in ['T0','T1','T2'])]
event_id = {val: idx+1 for idx, val in enumerate(wanted)}
```

An events array for MNE is created that contains the time of event occurrence in samples, placeholder 0 and class ID:

```python
events = np.array([
    [int(row['sample']), 0, event_id[row['value']]] 
    for _, row in events_df.iterrows() if row['value'] in event_id
])
```

Finally, the signal is segmented into epochs (0.5 to 2.5 seconds after the event), without baseline correction:

```python
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)
```

This step provides time intervals that are used for further analysis and classification.

### 2. Data and Events Overview

```python
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)
```

The following useful information is obtained:
- Number of epochs: 30
- Class TASK1T0: 15 epochs
- Class TASK1T1: 8 epochs
- Class TASK1T2: 7 epochs
- Used Annotations descriptions: ['T0', 'T1', 'T2']
- Event ID map: {'T0': 1, 'T1': 2, 'T2': 3}
- Used events: {'T0': 1, 'T1': 2}

### 3. CSP Analysis and Classification

Four variants of CSP analysis with different parameters (number of CSP components, classifier, frequency range) are defined and for each variant:
- epochs are filtered in the specified frequency range
- CSP components are calculated
- classifier (LDA or SVC) is trained using 5-fold cross-validation
- average accuracy and standard deviation are recorded

```python
params_variants = [
    {'n_csp': 4, 'clf': LDA(), 'freq': (7, 30),
     'desc': 'Original version: 4 CSP components, LDA classifier, 7-30 Hz'},
    {'n_csp': 6, 'clf': LDA(), 'freq': (7, 30),
     'desc': 'Variant 1: 6 CSP, LDA, 7-30 Hz'},
    {'n_csp': 4, 'clf': SVC(kernel="linear"), 'freq': (7, 30),
     'desc': 'Variant 2: 4 CSP, SVC (linear), 7-30 Hz'},
    {'n_csp': 4, 'clf': LDA(), 'freq': (8, 26),
     'desc': 'Variant 3: 4 CSP, LDA, 8-26 Hz'},
]
```

Running:
- Original version: 4 CSP components, LDA classifier, 7-30 Hz
  Accuracy: 0.790 ± 0.128
- Variant 1: 6 CSP, LDA, 7-30 Hz
  Accuracy: 0.830 ± 0.154
- Variant 2: 4 CSP, SVC (linear), 7-30 Hz
  Accuracy: 0.710 ± 0.156
- Variant 3: 4 CSP, LDA, 8-26 Hz
  Accuracy: 0.830 ± 0.154

### 4. Results Visualization

The graph below shows a comparison of accuracy for all CSP analysis variants. The graph clearly shows the variability in performance of different configurations, which helps in selecting optimal parameters.

### 5. Complete Code

```python
import mne
import pandas as pd
import numpy as np
from csp_analysis import run

# Part Two - real_data.py
edf_path = r"D:\IMR-PROJEKAT\ds004362\sourcedata\rawdata\S109\S109R03.edf"
events_path = r"D:\IMR-PROJEKAT\ds004362\sub-109\eeg\sub-109_task-motion_run-3_events.tsv"

# signal loading
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
# average reference
raw.set_eeg_reference('average', projection=True)
raw.filter(7., 30., fir_design='firwin', verbose=False)

# loading and filtering events for relevant classes (T0, T1, T2)
events_df = pd.read_csv(events_path, sep='\t')
wanted = [val for val in sorted(set(events_df['value'])) if any(t in val for t in ['T0','T1','T2'])]
# map that I later convert to array
event_id = {val: idx+1 for idx, val in enumerate(wanted)}

# events array
events = np.array([
    [int(row['sample']), 0, event_id[row['value']]] 
    for _, row in events_df.iterrows() if row['value'] in event_id
])

# creating epochs (0.5 to 2.5s after event)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)

print(f"Number of epochs: {len(epochs)}")
for k, v in event_id.items():
    print(f"Class {k}: {np.sum(epochs.events[:, 2] == v)} epochs")

# existing csp analysis/classification
run(raw)
```
