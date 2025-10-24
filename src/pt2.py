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
