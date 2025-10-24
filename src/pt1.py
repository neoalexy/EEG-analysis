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
