# Analiza EEG signala uz MNE-Python

Ova analiza prikazuje obradu i vizualizaciju EEG signala korišćenjem MNE-Python biblioteke. Cilj je demonstrirati osnovne korake u EEG obradi, uključujući filtriranje, segmentaciju (epohe), prosek (ERP), kao i poređenje signala pre i posle primene prosečne EEG reference. Posebna pažnja je posvećena različitim prikazima signala i njihovom značenju.

---

## 1. Vizualizacija senzora - Topomap ![](/pics/1.png)

Prikazujemo poziciju EEG senzora na površini glave u 2D topomap formatu. Ovaj prikaz je koristan za orijentaciju gde su elektroda smeštene, što je bitno za interpretaciju signala sa različitih lokacija mozga.

---

## 2. Vizualizacija senzora u 3D prostoru ![](/pics/2.png)

Ova slika prikazuje senzore u trodimenzionalnom prostoru, omogućavajući bolju prostornu predstavu pozicije elektroda u odnosu na oblik glave. Pomaže u razumevanju geometrije merenja.

---

## 3. Početni EEG signal bez prosečne reference ![](/pics/3.png)

Prikazuje sirovi EEG signal na prvih 5 kanala bez ikakve korekcije ili referentne promene. Ovo je osnovni prikaz kojim se može uočiti šum i varijacije u sirovom signalu.

---

## 4. EEG signal sa primenom prosečne EEG reference ![](/pics/4.png)

Nakon primene prosečne reference, signal je očišćeniji od šuma. Prosečna referenca podrazumeva oduzimanje prosečne vrednosti signala sa svih kanala od svakog pojedinačnog kanala, čime se dobija stabilniji i pouzdaniji EEG zapis.

---

## 5. Vizualizacija epoha ![](/pics/5.png)

Signal se segmentiše u epohe — vremenske intervale oko stimulusa. Ovo omogućava analizu moždanih odgovora koji su vremenski vezani za određene događaje. Slika prikazuje kako izgleda segmentisani EEG signal.

---

## 6. ERP signal za kanal eeg01 - auditory stimulus ![](/pics/6.png)

ERP (Event-Related Potential) predstavlja prosečan moždani odgovor na određeni stimulus. Ova slika prikazuje prosečan odgovor na auditivni stimulus („auditory/left“) na kanalu eeg01, sa prostornim bojama koje pomažu da se istaknu aktivni delovi.

---

## 7. ERP signal za kanal eeg01 - visual stimulus ![](/pics/7.png)

Poređenje sa prethodnom slikom — ERP odgovor na vizuelni stimulus („visual/left“) na istom kanalu, bez prostorne kolorne šeme, što može biti korisno za drugačiju interpretaciju signala.

---

## 8. Topomap raspored aktivnosti - početna verzija ![](/pics/8.png)

Prikazuje prostornu distribuciju moždane aktivnosti u određenim vremenskim momentima nakon stimulusa. Ova vizualizacija pomaže u lociranju aktivnih regiona mozga tokom različitih faza moždanog odgovora.

---

## 9. Topomap raspored aktivnosti - izmenjena verzija ![](/pics/9.png)

Slična prethodnoj, ali sa drugačijim podešavanjima (npr. bez kontura), što daje jasniji i drugačiji vizuelni efekat. Omogućava bolje razumevanje promena u prostornoj aktivnosti.

---

## 10. Joint prikaz ERP signala i topomap - početna verzija ![](/pics/10.png)

Kombinovani prikaz koji pokazuje kako se ERP oblik i prostorna aktivnost u mozgu menjaju u vremenu. Ova vizualizacija integriše dve dimenzije analize i olakšava dublje razumevanje moždanih odgovora.

---

## 11. Joint prikaz ERP signala i topomap - izmenjena verzija ![](/pics/11.png)

Isti koncept kao i na prethodnoj slici, ali sa modifikovanim vremenskim intervalima i parametrima za bolju jasnoću i različite uvide u podatke.

---

## 12. Power Spectral Density (PSD) - početna verzija ![](/pics/12.png)

Prikaz raspodele snage signala po frekvencijama za kanal eeg01. Ovo pomaže u identifikaciji dominantnih frekvencija, kao što su alfa, beta, gama talasi, i može ukazivati na potencijalne artefakte ili specifične moždane aktivnosti.

---

## 13. PSD analiza za kanal eeg02 - druga verzija ![](/pics/13.png)

Slična analiza kao prethodna, ali za drugi kanal i sa drugačijim podešavanjima (npr. bez proseka), što može otkriti dodatne informacije o frekvencijskoj strukturi signala.

---

## 14. Heatmap signala eeg01 sa manjim sigma ![](/pics/14.png)

Heatmap prikazuje snagu signala kroz vreme i frekvenciju, gde je sigma parametar koji kontroliše glatkoću prikaza. Manji sigma daje detaljniji i oštriji prikaz, ali sa više šuma.

---

## 15. Heatmap signala eeg01 sa većim sigma ![](/pics/15.png)

Veći sigma daje glatkiji, „mekši“ prikaz signala, pomažući da se uoče veći obrasci i smanjuje šum, ali može izostaviti fine detalje.

---

## 16. Poređenje ERP signala auditory i visual stimulusa ![](/pics/16.png)

Na ovoj slici je prikazano poređenje ERP signala sa dva različita tipa stimulusa na kanalu eeg01. Vidljive su razlike u amplitudama i vremenskim obrascima, što ukazuje na različite moždane procese aktivirane auditivnim i vizuelnim stimulusom.

---

# Zaključak

Ova analiza demonstrira osnovne tehnike u obradi EEG signala, sa naglaskom na vizualizaciju i poređenje podataka pre i posle različitih koraka obrade. Posebno je istaknuta uloga prosečne reference u čišćenju signala, kao i značaj višestrukih vrsta vizualizacija (topomap, ERP, PSD, heatmap) za razumevanje složenih moždanih odgovora.

Ove tehnike predstavljaju temelj za naprednije EEG analize i mogu se koristiti u neuroznanosti, kliničkoj dijagnostici i razvoju BCI (Brain-Computer Interface) sistema.
