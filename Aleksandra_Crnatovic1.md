# Analiza i prikaz EEG signala - MNE-Python
##Prvi deo - prvi primer

Ova analiza prikazuje obradu i vizualizaciju EEG signala koriscenjem MNE-Python biblioteke sa ciljem demonstracije osnovnih koraka u obradi EEG signala ukljucujuci filtriranje, segmentaciju (epohe), prosek (ERP). Radi boljeg upoznavanja sa promenama parametara u analizi EEG signala, prikazana su poredjenja slicnih talasa sa nekim vrstama izmena.

---

## 1. Vizualizacija senzora - Topomap 
![](/pics/1.png)

Prvenstveno prikazujem poziciju EEG senzora na povrsini glave u 2D formatu. Ovaj prikaz je koristan za orijentaciju gde su smestene elektrode, sto je od presudne vaznosti za interpretaciju signala sa razlicitih lokacija mozga.
```python 
raw.plot_sensors(kind='topomap', show_names=True)
```
---

## 2. Vizualizacija senzora - 3D prikaz
![](/pics/2.png)

Ova slika prikazuje senzore u trodimenzionalnom prostoru sto nam omogucava bolju prostornu predstavu pozicije elektroda u odnosu na sam oblik glave i takodje pomaze u razumevanju geometrije merenja.
```python 
raw.plot_sensors(kind='3d')
```
---

## 3. Pocetni EEG signal bez prosecne reference 
![](/pics/3.png)

Prikazan je sirovi EEG signal na prvih 5 kanala bez ikakvih korekcija ili promena. Ovo je osnovni prikaz na kojem se moze uociti sum i ostale varijacije.
```python 
raw.plot(n_channels=5, proj=False)
```
---

## 4. EEG signal sa primenom prosecne EEG reference 
![](/pics/4.png)

Nakon primene prosecne reference signal je cistiji i stabilniji. Prosecna referenca podrazumeva oduzimanje prosecne vrednosti signala sa svih kanala od svakog pojedinacnog kanala cime se dobija stabilniji i pouzdaniji EEG zapis.
```python 
raw.set_eeg_reference('average')
raw.plot(n_channels=5, proj=True)
```

---

## 5. Vizualizacija epoha 
![](/pics/5.png)

Signal se segmentise u epohe koje predstavljaju vremenske intervale oko stimulusa. Ovo omogucava analizu mozdanih odgovora koji su vremenski vezani za odredjene dogadjaje. Slika prikazuje kako izgleda segmentisani EEG signal.
```python 
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7, preload=True)
epochs.drop_bad(reject=dict(eeg=100e-6, eog=21e-6))
epochs.plot()
```
---

## 6. ERP signal za kanal eeg01 - auditory stimulus 
![](/pics/6.png)

ERP (Event-Related Potential) predstavlja prosecan mozdani odgovor na odredjeni stimulus. Ova slika prikazuje prosecan odgovor na auditivni stimulus („auditory/left“) na kanalu eeg01. Prostorne boje pomazu da se istaknu aktivni delovi.
```python 
erp1 = epochs['auditory/left'].average()
erp1.plot(picks="eeg01", spatial_colors=True)
```

---

## 7. ERP signal za kanal eeg01 - visual stimulus 
![](/pics/7.png)

Poredjenje sa prethodnim prikazom — ERP odgovor na vizuelni stimulus („visual/left“) na istom kanalu, ovog puta bez prostorne kolorne seme sto moze biti korisno za drugaciju interpretaciju signala.
```python 
erp2 = epochs['visual/left'].average()
erp2.plot(picks="eeg01", spatial_colors=False)
```

---

## 8. Topomap raspored aktivnosti 
![](/pics/8.png)

Prikazuje prostornu distribuciju mozdane aktivnosti u odredjenim vremenskim momentima nakon stimulusa. Ovom vizualizacijom se dobija uvid u lociranju aktivnih regiona mozga tokom razlicitih faza mozdanog odgovora.
```python 
fig1 = erp1.plot_topomap(times=[0.1, 0.2, 0.3])
fig1.suptitle("Topomap - Pocetna verzija (auditory/left)")
```
---

## 9. Topomap raspored aktivnosti - izmenjena verzija 
![](/pics/9.png)

Slicna prethodnoj ali sa drugacijim parametrima (bez kontura). Omogucava bolje razumevanje promena u prostornoj aktivnosti.
```python 
fig2 = erp2.plot_topomap(times=[0.1, 0.2, 0.3], contours=0)
fig2.suptitle("Topomap - Druga verzija (visual/left)")
```

---

## 10. Joint prikaz ERP signala i topomap 
![](/pics/10.png)     ![](/pics/11.png)

Na slici gore - kombinovani prikaz koji pokazuje kako se ERP oblik i prostorna aktivnost u mozgu menjaju u vremenu. Na slici dole - slican koncept kao i na prethodnoj slici ali sa modifikovanim vremenskim intervalima i parametrima za bolju jasnocu i razlicite uvide u podatke.
```python 
erp1.plot_joint(title="ERP + Topomap - Pocetna verzija (auditory/left)", times=[0.1, 0.2])
erp2.plot_joint(title="ERP + Topomap - Druga verzija (visual/left)", times=[0.2, 0.3])
```

---


## 12. Power Spectral Density (PSD)
![](/pics/12.png)

Prikaz raspodele snage signala po frekvencijama za kanal eeg01. Ovako se olaksava identifikovanje dominantnih frekvencija tj. alfa, beta, gama talasi, a moze i ukazivati na potencijalne artefakte ili specificne mozdane aktivnosti.
```python 
psd1 = epochs['auditory/left'].compute_psd(fmin=1, fmax=40)
psd1.plot(picks="eeg01", average=True, spatial_colors=True)
```
---

## 13. PSD analiza za kanal eeg02
![](/pics/13.png)

Slicna analiza kao prethodna ali za drugi kanal (eeg02) i sa za nijansu drugacijim parametrima (bez proseka).
```python 
psd1.plot(picks="eeg02", average=False, spatial_colors=False)
```
---

## 14. Heatmap signala eeg01
![](/pics/14.png)

Heatmap prikazuje snagu signala kroz vreme i frekvenciju gde je sigma parametar koji odredjuje glatkocu prikaza. Manji sigma daje detaljniji i ostriji prikaz ali sa više suma.
```python 
epochs['auditory/left'].plot_image(picks="eeg01", sigma=0.1, title="Heatmap eeg01 - Pocetna verzija")
```
---

## 15. Heatmap signala eeg01 sa većim sigma 
![](/pics/15.png)

Povecanjem sigma parametra dobija se „meksi“ prikaz signala, cime se olaksava uocavanje vecih obrazaca i smanjuje sum ali ipak moze izostaviti fine detalje.
```python 
epochs['auditory/left'].plot_image(picks="eeg01", sigma=0.3, title="Heatmap eeg01 - Druga verzija")
```
---

## 16. Poređenje ERP signala auditory i visual stimulusa 
![](/pics/16.png)

Na ovoj slici je prikazano poređenje ERP signala sa dva razlicita tipa stimulusa na kanalu eeg01. Vidljive su razlike u amplitudama i vremenskim obrascima sto ukazuje na razlicite mozdane procese aktivirane auditivnim i vizuelnim stimulusom.
```python 
plt.plot(erp1.times, erp1.data[erp1.ch_names.index("eeg01")] * 1e6, label='Auditory Left')
plt.plot(erp2.times, erp2.data[erp2.ch_names.index("eeg01")] * 1e6, label='Visual Left', linestyle='--')
```
---

# Zakljucak

Ova analiza demonstrira osnovne tehnike u obradi EEG signala sa naglaskom na vizualizaciju i poredjenje podataka pre i posle razlicitih koraka obrade. Primenjene su visestruke vrste vizualizacija (topomap, ERP, PSD, heatmap).


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

