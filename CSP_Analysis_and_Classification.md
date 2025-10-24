## Prvi deo - Drugi primer - Motorna imaginacija: CSP analiza i klasifikacija

U ovom delu se koristi metoda **Common Spatial Patterns (CSP)** za ekstrakciju prostorno-frekvencijskih obrazaca iz EEG signala vezanih za motorne imaginacione zadatke. Nakon ekstrakcije karakteristika koristi se klasifikacija za razlikovanje stanja uma.

### 1. Ucitavanje i priprema podataka

Podaci za subjekta 1 se ucitavaju za dve sesije koje sadrze motor imaginarne zadatke (runs 6 i 10). Zatim se podaci filtriraju u frekvencijskom opsegu od 7 do 30 Hz radi izdvajanja relevantnih oscilacija, i segmentišu u epohe.
```python
raw_files = [eegbci.load_data(subject, run) for run in runs]
raws = [read_raw_edf(f[0], preload=True, verbose=False) for f in raw_files]
raw = concatenate_raws(raws)

raw.pick_types(eeg=True, exclude='bads')  # Izbor EEG kanala
raw.set_eeg_reference('average', projection=True)  # Prosecna EEG referenca
raw.filter(7., 30., fir_design='firwin', verbose=False)  # Band-pass filter

events, event_id = mne.events_from_annotations(raw)  # Dogadjaji i njihova ID mapa
epochs = mne.Epochs(raw, events, event_id=wanted_events, tmin=0.5, tmax=2.5,
                    baseline=None, preload=True, verbose=False)  # Segmentacija u epohe
```
Filterom se zadržavaju oscilacije u beta i alfa opsegu (važno za motornu imaginaciju).
Epohe predstavljaju vremenske intervale oko stimulusa koje koristimo za analizu.

### 2. Definisanje parametara eksperimenata

Eksperiment se izvodi u cetiri varijante gde se menjaju parametri:

- Broj CSP komponenti (4 ili 6)
- Klasifikator (LDA ili SVM sa linearnim kernelom)
- Frekvencijski opseg (7-30 Hz ili 8-26 Hz)
```python
params_variants = [
    {'n_csp': 4, 'clf': LDA(), 'freq': (7, 30), 'desc': 'Originalna verzija'},
    {'n_csp': 6, 'clf': LDA(), 'freq': (7, 30), 'desc': 'Varijanta 1'},
    {'n_csp': 4, 'clf': SVC(kernel="linear"), 'freq': (7, 30), 'desc': 'Varijanta 2'},
    {'n_csp': 4, 'clf': LDA(), 'freq': (8, 26), 'desc': 'Varijanta 3'},
]
```
Ove varijante omogucavaju procenu uticaja parametara na performanse klasifikacije.
Promenom broja CSP komponenti menjamo koliko prostorno-diskriminativnih karakteristika izvučemo. Različiti klasifikatori i frekvencijski opsezi mogu uticati na tačnost.

### 3. Pokretanje CSP analize i klasifikacije

Za svaku varijantu:

- Filtriraju se podaci po zadatom frekvencijskom opsegu
- Izracunavaju se CSP komponente koje naglasavaju razlike izmedju klasa
- Klasifikator se trenira i ocenjuje koristeci 5-fold cross-validaciju
- Izracunava se prosecna tacnost i standardna devijacija

```python
epochs_filtered = epochs.copy().filter(freq[0], freq[1], fir_design='firwin', verbose=False)
X = epochs_filtered.get_data()  # Podaci u obliku (epohe, kanali, vreme)
y = labels

csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
clf_pipe = Pipeline([('CSP', csp), ('Classifier', clf)])

scores = cross_val_score(clf_pipe, X, y, cv=5, n_jobs=1)
mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"Tačnost: {mean_score:.3f} ± {std_score:.3f}")
```
CSP projekcija ističe razlike između klasa.
Pipeline kombinuje CSP i klasifikator u jednu celinu.
Cross-validation daje procenu generalizacije modela.

### 4. Rezultati i poredjenje

| Varijanta   | Tacnost (srednja ± std) |
|-------------|-------------------------|
| Originalna  |     0.842 ± 0.112       |
| Varijanta 1 |     0.794 ± 0.085       |
| Varijanta 2 |     0.728 ± 0.087       |
| Varijanta 3 |     0.864 ± 0.043       |

Najbolji rezultat postignut kod Varijante 3 sa 4 CSP komponente, LDA klasifikatorom i frekvencijskim opsegom 8-26 Hz.

### 5. Vizualizacija poredenja

![Poređenje tacnosti razlicitih varijanti CSP analize](/pics/grafikon_bar.png)

Grafikon jasno pokazuje koja varijanta je dala najbolje rezultate i varijabilnost tačnosti kroz standardnu devijaciju.

### Zakljucak

Ova analiza pokazuje kako izbor parametara CSP metode i klasifikatora utiče na preciznost dekodiranja motoricke imaginacije iz EEG signala. 
CSP efikasno istice diskriminativne prostorne obrasce a LDA klasifikator daje bolje performanse u ovom slucaju u odnosu na SVM sa linearnim kernelom. 
Pored toga, frekvencijski opseg signala je znacajan faktor za optimizaciju rezultata.

---
