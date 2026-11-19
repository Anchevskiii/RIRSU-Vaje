# Klasifikacija znakov kamen, škarje, papir s konvolucijskimi nevronskimi mrežami (CNN)

Dokument predstavlja tehnični pregled priprave podatkov, postopnega eksperimentalnega razvoja konvolucijske nevronske mreže od začetka (*from scratch*) ter priprave napovedi za slike znakov **Paper**, **Rock** in **Scissors**. Celotna koda, grafi in izpisi se nahajajo v zvezku [rps_cnn.ipynb](rps_cnn.ipynb).

---

## 1. Pregled podatkovne zbirke

Podatkovna zbirka v mapi `rirsu-2026-klasifikacija/` je sestavljena iz dveh delov:
- **Učna zbirka (`train/`, skupaj 3.150 sivinskih slik JPG)**:
  - `Paper`: 1.028 slik
  - `Rock`: 1.050 slik
  - `Scissors`: 1.072 slik
- **Testna zbirka (`test/`, skupaj 1.350 sivinskih slik JPG)**: slike od `test_image_0.jpg` do `test_image_1349.jpg` brez oznak razredov.

Slike imajo različna razmerja stranic (najpogosteje `100x56`, `75x100`, `100x100`, `56x100` in `100x176`) ter zelo raznolika realna ozadja in pogoje osvetlitve. Testne slike se med razvojem, učenjem in izbiro modela ne uporabljajo.

---

## 2. Predobdelava, delitev podatkov in augmentacija

### 2.1 Delitev in cevovod `tf.data.Dataset`
- Učna zbirka se stratificirano razdeli v razmerju **80 / 20** (`SEED = 1234`) na **2.520 učnih** in **630 validacijskih** slik.
- Konsistentna številska preslikava razredov: `{'Paper': 0, 'Rock': 1, 'Scissors': 2}`.
- Vsaka slika se v cevovodu `tf.data.Dataset` prebere z `tf.io.decode_jpeg(..., channels=1)` kot enokanalna sivinska slika, z `tf.image.resize_with_pad(..., 80, 80)` prilagodi na enotno dimenzijo $80 \times 80 \times 1$ brez popačenja oblike prstov in normalizira na interval `[0, 1]` z deljenjem z `255.0`.
- Uporaba `.cache()` po predobdelavi shrani pripravljene tenzorje v pomnilnik, `.shuffle(..., reshuffle_each_iteration=True)` vsako epoho premeša samo učne primere, `.batch(32)` in `.prefetch(tf.data.AUTOTUNE)` pa pospešita učenje.

### 2.2 Podatkovna augmentacija (samo med učenjem)
V model je vključen blok `Sequential` s petimi naključnimi transformacijami, ki se izvajajo izključno med učenjem (`training=True`) in ohranjajo pravilen razred:
1. `RandomFlip('horizontal')` – posnema levo ali desno roko (navpično zrcaljenje ni uporabljeno, saj so zapestja običajno usmerjena navzgor).
2. `RandomRotation(0.12, fill_mode='nearest')` – manjši nagibi zapestja.
3. `RandomZoom(0.12, fill_mode='nearest')` – različna oddaljenost roke od kamere.
4. `RandomTranslation(0.08, 0.08, fill_mode='nearest')` – različni položaji dlani v kadru.
5. `RandomContrast(0.2, value_range=(0, 1))` – prilagoditev razlikam v osvetlitvi in kontrastu znotraj intervala `[0, 1]`.

Nastavitev `fill_mode='nearest'` pri rotaciji, povečavi in premiku podaljša robne piksle slike namesto vstavljanja ostrih črnih robov.

---

## 3. Eksperimenti in primerjava rezultatov

Vsi štirje modeli so zgrajeni in naučeni od začetka iz standardnih slojev knjižnice Keras z največ 100 epohami ter ovrednoteni na enaki validacijski množici (630 slik) s funkcijo izgube `sparse_categorical_crossentropy`.

| Eksperiment | Ključne spremembe | Parametri | Epoh | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Exp 1: Osnovni CNN** | 3 bloki `Conv2D(32/64/128) + MaxPool`, `Flatten`, `Dense(128)`, `Adam(1e-3)` | 1.731.587 | 22 | 0,9873 | 0,9875 | 0,9873 | 0,9873 |
| **Exp 2: Osnovni + augmentacija** | Enaka arhitektura + podatkovna augmentacija med učenjem | 1.731.587 | 58 | 0,9905 | 0,9905 | 0,9905 | 0,9905 |
| **Exp 3: 4-blokovni BN + GAP + Reg** | 4 bloki (`32->64->128->256`), `BatchNorm`, `GlobalAveragePooling2D`, `Dropout(0.4/0.3)`, `L2(1e-4)`, `ReduceLROnPlateau` | 422.563 | 66 | 0,9429 | 0,9455 | 0,9429 | 0,9429 |
| **Exp 4: Dvojni bloki + `AdamW`** | Dvojni sloji `Conv2D(3x3)` v blokih 2–4, `SpatialDropout2D(0.1/0.15)`, ohranitev `10x10` pred `GAP`, `AdamW(lr=1e-3, wd=1e-4)` | **1.198.499** | **80** | **0,9968** | **0,9968** | **0,9968** | **0,9968** |

### Analiza vpliva sprememb
- **Vpliv augmentacije (Exp 1 vs. Exp 2)**: V Eksperimentu 1 učna točnost že okoli 10. epohe doseže `1,000`, validacijska izguba pa se ustavi pri ~`0,035–0,045`, kar kaže na hitro pomnjenje učnih slik. Dodana augmentacija v Eksperimentu 2 upočasni pomnjenje, približa učno in validacijsko krivuljo ter dvigne `Weighted F1` z `0,9873` na `0,9905`.
- **Vpliv arhitekture in regularizacije (Exp 3 vs. Exp 4)**: V Eksperimentu 3 so štiri zaporedne plasti `MaxPooling2D` skrčile mapo značilk na `5 x 5`, ob tem pa je le ena konvolucija na blok skupaj z močnim `Dropout(0.4/0.3)` omejila kapaciteto modela (`0,9429`). V Eksperimentu 4 smo to odpravili z dvojnimi `3 x 3` konvolucijami v globljih blokih, ohranitvijo prostorske ločljivosti `10 x 10` pred `GlobalAveragePooling2D`, prostorskim `SpatialDropout2D` in optimizatorjem `AdamW`.
- **Izbira končnega modela**: Izbran je **Eksperiment 4**, ki doseže najboljši validacijski rezultat **`Weighted F1 = 0,9968`** (628/630 pravilno razvrščenih validacijskih slik: `Paper` 206/206, `Rock` 209/210, `Scissors` 213/214) ob ~31 % manjšem številu parametrov kot Eksperiment 1 in 2.

---

## 4. Shranjeni model in ponovno generiranje napovedi

Končni model je shranjen v datoteki `best_rps_cnn.keras`, končne napovedi za vseh 1.350 testnih slik pa v datoteki `submission.csv` v obliki:

```csv
image,predicted_class
test_image_0.jpg,Rock
test_image_1.jpg,Scissors
test_image_2.jpg,Rock
```

Za hitro ponovno generiranje datoteke `submission.csv` iz shranjenega modela brez ponovnega učenja zadostuje naslednji postopek v mapi z zvezkom:

```python
import os, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 80
idx_to_label = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}
model = keras.models.load_model('best_rps_cnn.keras')

test_dir = os.path.join('rirsu-2026-klasifikacija', 'test')
test_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')],
    key=lambda f: int(re.search(r'(\d+)', f).group(1)),
)
test_paths = np.array([os.path.join(test_dir, f) for f in test_files])

def load_test_image(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)
    return tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE) / 255.0

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

preds = np.argmax(model.predict(test_ds), axis=1)
pd.DataFrame({
    'image': test_files,
    'predicted_class': [idx_to_label[i] for i in preds],
}).to_csv('submission.csv', index=False)
```

---

## 5. Ključna vprašanja za predstavitev

**V: Zakaj smo uporabili `resize_with_pad` namesto navadnega `resize`?**
> **O**: Slike v zbirki imajo različna razmerja stranic (npr. `100x56` in `56x100`). Navadno raztezanje v kvadrat $80 \times 80$ bi močno popačilo obliko dlani in prstov, `resize_with_pad` pa ohrani izvirno geometrijsko razmerje ter doda robno polnilo.

**V: Zakaj je Eksperiment 4 bistveno presegel Eksperiment 3 kljub podobni regularizaciji?**
> **O**: Eksperiment 3 je imel na vsakem nivoju le en konvolucijski sloj in štirikratno združevanje `MaxPooling2D` (ločljivost pred `GAP` je padla na $5 \times 5$). Eksperiment 4 z dvojnimi `3 x 3` konvolucijami poveča efektivno receptivno polje, pred `GlobalAveragePooling2D` ohrani večjo prostorsko mapo ($10 \times 10 \times 256$) ter uporablja `SpatialDropout2D`, ki izklaplja celotne kanale značilk namesto posameznih pikslov.

**V: Zakaj pri ponovnem nalaganju `best_rps_cnn.keras` ni treba ročno izklapljati augmentacije ali `Dropout` slojev?**
> **O**: Tako augmentacijski sloji kot `Dropout`, `SpatialDropout2D` in `BatchNormalization` v Kerasu samodejno preklopijo na način sklepanja (`training=False`), ko kličemo `model.predict(...)` ali `validation_data`. Med testiranjem se torej slike ne spreminjajo, za `BatchNormalization` pa se uporabijo naučene drseče statistike.

---

## 6. Hitri pregled pojmov in parametrov (kaj, kje, zakaj, učinek)

| Pojem / parameter | Kaj počne | Kje in zakaj je uporabljen | Konkretni učinek v modelu |
|---|---|---|---|
| `resize_with_pad(80, 80)` | Prilagodi slike različnih razmerij na `80x80` z robnim polnilom | Predobdelava v `load_image` namesto raztezanja, da oblika prstov ostane nespremenjena | Enoten vhod `(80, 80, 1)` brez popačenja dlani |
| `.cache()` v `tf.data` | Shrani enkrat dekodirane in pomanjšane slike v pomnilnik | V `make_dataset` takoj za `map(load_image)` pred mešanjem | Bistveno pospeši vse nadaljnje epohe na CPE |
| Augmentacija (`nearest`, `(0, 1)`) | Vodoravno zrcali, vrti, poveča, premika in prilagodi kontrast | V modelih Exp 2–4 (samo pri `training=True`); `nearest` in `(0, 1)` preprečita ostre črne robove | Zmanjša pomnjenje učnih slik; F1 naraste z `0,9873` (Exp 1) na `0,9905` (Exp 2) |
| `BatchNormalization` (`use_bias=False`) | Standardizira aktivacije za vsakim `Conv2D` na podobno merilo | V blokih modelov Exp 3 in 4 (zaradi različne osvetlitve fotografij; `use_bias=False`, ker premik doda BN) | Stabilizira in omogoči učenje globljih konvolucijskih blokov |
| `GlobalAveragePooling2D` (`GAP`) | Vsako zadnjo mapo značilk povpreči v eno samo število (`256` vrednosti) | V Exp 3 in 4 namesto `Flatten` | Močno zmanjša število parametrov glave in poveča odpornost na položaj dlani |
| `SpatialDropout2D(0.1 / 0.15)` | Med učenjem naključno izklopi celotne 2D kanale značilk | V blokih 2 in 3 modela Exp 4 namesto navadnega 2D točkovnega izklapljanja | Prepreči zanašanje na posamezne filtre ob visoki prostorski korelaciji |
| `Dropout(0.4/0.3)` + `L2(1e-4)` | Izklopi delež nevronov in kaznuje velike uteži v sloju `Dense(128)` | V klasifikacijski glavi modelov Exp 3 in 4 | Omeji preprileganje v gostem sloju |
| `ReduceLROnPlateau` | Prepolovi učno stopnjo (`factor=0.5`), ko `val_loss` miruje 5–6 epoh | V Exp 3 in 4 poleg `EarlyStopping(patience=15)` | Umiri nihanja na validaciji in omogoči natančnejšo konvergenco |
| `AdamW(lr=1e-3, wd=1e-4)` | Optimizator `Adam` z ločenim upadanjenjem uteži (*weight decay*) | V končnem modelu Exp 4 | Izboljša posplošitev globlje mreže do `0,9968` |
| Dvojni `Conv2D(3x3)` + `10x10` pred `GAP` | Dva zaporedna sloja `3x3` (efektivno `5x5`) in samo 3× `MaxPooling2D` | V blokih 2–4 modela Exp 4 (odprava ozkega grla iz Exp 3, kjer je ostalo le `5x5`) | Dvigne validacijski Weighted F1 z `0,9429` (Exp 3) na **`0,9968`** (628/630 pravilnih) |
