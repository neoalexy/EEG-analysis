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


