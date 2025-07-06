## Drugi deo - Realni podaci: Ucitavanje, segmentacija i CSP analiza sa klasifikacijom

U ovom delu se radi ucitavanje stvarnih EEG podataka iz EDF fajlova i eventova iz TSV fajlova. Podaci se filtriraju, segmentisu u epohe po zadatim eventovima a nakon toga se primenjuje vec postojeca CSP analiza (PrviDeo_1) i klasifikacija za razlikovanje motorickih zadataka.

### 1. Ucitavanje i priprema podataka

Prvo se ucitava EEG signal iz EDF fajla a zatim se primenjuje prosecna referenca i filtriranje u frekvencijskom opsegu od 7 do 30 Hz da bi se izdvojile znacajne oscilacije.

```python
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
raw.set_eeg_reference('average', projection=True)
raw.filter(7., 30., fir_design='firwin', verbose=False)
```
Zatim se ucitava TSV fajl sa eventovima i filtriramo eventove koji su relevantni za zadatak (npr. oznake T0, T1, T2), i definise se mapa event_id koja svakoj klasi dodeljuje jedinstveni broj.

```python
events_df = pd.read_csv(events_tsv_path, sep='\t')
wanted = [val for val in sorted(set(events_df['value'])) if any(t in val for t in ['T0','T1','T2'])]
event_id = {val: idx+1 for idx, val in enumerate(wanted)}
```
Kreira se events niz za MNE koji sadrzi vreme pojave eventa u uzorcima, placeholder 0 i ID klase:
```python
events = np.array([
    [int(row['sample']), 0, event_id[row['value']]] 
    for _, row in events_df.iterrows() if row['value'] in event_id
])
```
Na kraju, segmentira se signal u epohe (0.5 do 2.5 sekundi nakon eventa), bez baseline korekcije:
```python
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)

```
Ovaj korak pruza vremenske intervale koji se koriste za dalju analizu i klasifikaciju.

### 2. Pregled podataka i eventova
```python
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)
```
Dobijaju se odgovarajuce korisne informacije: 
Broj epoha: 30
Klasa TASK1T0: 15 epoha
Klasa TASK1T1: 8 epoha
Klasa TASK1T2: 7 epoha
Used Annotations descriptions: ['T0', 'T1', 'T2']
Event ID mapa: {'T0': 1, 'T1': 2, 'T2': 3}
Korišćeni događaji: {'T0': 1, 'T1': 2}

### 3.  CSP analiza i klasifikacija
Definisane su 4 varijante CSP analize sa razlicitim parametrima (broj CSP komponenti, klasifikator, frekvencijski opseg) i za svaku varijantu:
-filtriraju se epohe u zadatom frekvencijskom opsegu
-izracunavaju se CSP komponente
-trenira se klasifikator (LDA ili SVC) koristeći 5-fold cross-validaciju
-belezi se prosecna tacnost i standardna devijacija
```python
params_variants = [
    {'n_csp': 4, 'clf': LDA(), 'freq': (7, 30),
     'desc': 'Originalna verzija: 4 CSP komponente, LDA klasifikator, 7-30 Hz'},
    {'n_csp': 6, 'clf': LDA(), 'freq': (7, 30),
     'desc': 'Varijanta 1: 6 CSP, LDA, 7-30 Hz'},
    {'n_csp': 4, 'clf': SVC(kernel="linear"), 'freq': (7, 30),
     'desc': 'Varijanta 2: 4 CSP, SVC (linear), 7-30 Hz'},
    {'n_csp': 4, 'clf': LDA(), 'freq': (8, 26),
     'desc': 'Varijanta 3: 4 CSP, LDA, 8-26 Hz'},
]

```
Pokrecem: Originalna verzija: 4 CSP komponente, LDA klasifikator, 7-30 Hz
Tacnost: 0.790 ± 0.128
Pokrecem: Varijanta 1: 6 CSP, LDA, 7-30 Hz
Tacnost: 0.830 ± 0.154
Pokrecem: Varijanta 2: 4 CSP, SVC (linear), 7-30 Hz
Tacnost: 0.710 ± 0.156
Pokrecem: Varijanta 3: 4 CSP, LDA, 8-26 Hz
Tacnost: 0.830 ± 0.154

### 4. Vizualizacija rezultata
Na grafikonu ispod prikazano je poredjenje tacnosti svih varijanti CSP analize.
Grafikon jasno pokazuje varijabilnost performansi razlicitih konfiguracija sto pomaze u odabiru optimalnih parametara.
![](/pics/realni_podaci_grafikon.png)

### 5. Ceo kod
```python
import mne
import pandas as pd
import numpy as np
from csp_analysis import run
#Drugi deo-rwal_data.py
edf_path = r"D:\IMR-PROJEKAT\ds004362\sourcedata\rawdata\S109\S109R03.edf"
events_path = r"D:\IMR-PROJEKAT\ds004362\sub-109\eeg\sub-109_task-motion_run-3_events.tsv"

#ucitavanje signala
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
#prosecna refernca
raw.set_eeg_reference('average', projection=True)
raw.filter(7., 30., fir_design='firwin', verbose=False)

# ucitavanje i filtriranje eventova za relevantne klase (T0, T1, T2)
events_df = pd.read_csv(events_path, sep='\t')
wanted = [val for val in sorted(set(events_df['value'])) if any(t in val for t in ['T0','T1','T2'])]
#mapa koju kasnij ekonverujem u niz 
event_id = {val: idx+1 for idx, val in enumerate(wanted)}

#events niz
events = np.array([
    [int(row['sample']), 0, event_id[row['value']]] 
    for _, row in events_df.iterrows() if row['value'] in event_id
])

#kreiranje epoha (0.5 do 2.5s nakon eventa)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)

print(f"Broj epoha: {len(epochs)}")
for k, v in event_id.items():
    print(f"Klasa {k}: {np.sum(epochs.events[:, 2] == v)} epoha")

#postojeca csp analiza/klasifikacija
run(raw)

```







