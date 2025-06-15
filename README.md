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

Na slici levo - kombinovani prikaz koji pokazuje kako se ERP oblik i prostorna aktivnost u mozgu menjaju u vremenu. Na slici desno - slican koncept kao i na prethodnoj slici ali sa modifikovanim vremenskim intervalima i parametrima za bolju jasnocu i razlicite uvide u podatke.
```python 
erp1.plot_joint(title="ERP + Topomap - Pocetna verzija (auditory/left)", times=[0.1, 0.2])
erp2.plot_joint(title="ERP + Topomap - Druga verzija (visual/left)", times=[0.2, 0.3])
```

---


## 12. Power Spectral Density (PSD) - početna verzija 
![](/pics/12.png)

Prikaz raspodele snage signala po frekvencijama za kanal eeg01. Ovo pomaže u identifikaciji dominantnih frekvencija, kao što su alfa, beta, gama talasi, i može ukazivati na potencijalne artefakte ili specifične moždane aktivnosti.

---

## 13. PSD analiza za kanal eeg02 - druga verzija 
![](/pics/13.png)

Slična analiza kao prethodna, ali za drugi kanal i sa drugačijim podešavanjima (npr. bez proseka), što može otkriti dodatne informacije o frekvencijskoj strukturi signala.

---

## 14. Heatmap signala eeg01 sa manjim sigma 
![](/pics/14.png)

Heatmap prikazuje snagu signala kroz vreme i frekvenciju, gde je sigma parametar koji kontroliše glatkoću prikaza. Manji sigma daje detaljniji i oštriji prikaz, ali sa više šuma.

---

## 15. Heatmap signala eeg01 sa većim sigma 
![](/pics/15.png)

Veći sigma daje glatkiji, „mekši“ prikaz signala, pomažući da se uoče veći obrasci i smanjuje šum, ali može izostaviti fine detalje.

---

## 16. Poređenje ERP signala auditory i visual stimulusa 
![](/pics/16.png)

Na ovoj slici je prikazano poređenje ERP signala sa dva različita tipa stimulusa na kanalu eeg01. Vidljive su razlike u amplitudama i vremenskim obrascima, što ukazuje na različite moždane procese aktivirane auditivnim i vizuelnim stimulusom.

---

# Zaključak

Ova analiza demonstrira osnovne tehnike u obradi EEG signala, sa naglaskom na vizualizaciju i poređenje podataka pre i posle različitih koraka obrade. Posebno je istaknuta uloga prosečne reference u čišćenju signala, kao i značaj višestrukih vrsta vizualizacija (topomap, ERP, PSD, heatmap) za razumevanje složenih moždanih odgovora.

Ove tehnike predstavljaju temelj za naprednije EEG analize i mogu se koristiti u neuroznanosti, kliničkoj dijagnostici i razvoju BCI (Brain-Computer Interface) sistema.
