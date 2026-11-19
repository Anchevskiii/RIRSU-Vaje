# Klasifikacija geometrijskih oblik s konvolucijskimi nevronskimi mrežami (CNN)

Dokument predstavlja podroben tehnični vodnik skozi postopek priprave slikovnih podatkov, gradnje, učenja in vrednotenja konvolucijskih nevronskih mrež za prepoznavanje geometrijskih oblik. Celotna izvedljiva koda z vsemi izpisi, tabelami in grafi se nahaja v zvezku [naloga4.ipynb](naloga4.ipynb).

---

## 1. Pregled problema in podatkovne zbirke

Cilj problema je razvrščanje sivinskih slik preprostih ročno narisanih geometrijskih likov v tri uravnotežene razrede: **krog (`circle`)**, **kvadrat (`square`)** in **trikotnik (`triangle`)**.

Podatkovna zbirka v mapi `shapes/` vsebuje skupaj **300 slik formata PNG** (po 100 slik v podmapah `circles/`, `squares/` in `triangles/`). Slike so velikosti $28 \times 28$ slikovnih pik, narisane s temno črto na belem ozadju (vrednosti pikslov blizu `255`). Za razliko od klasičnih modelov (npr. odločitvenega drevesa, KNN ali večplastnega perceptrona MLP), ki sliko sploščijo v enodimenzionalni vektor dolžine 784 in pri tem izgubijo prostorsko soseščino pikslov, konvolucijska nevronska mreža (CNN) neposredno obdeluje dvodimenzionalno matriko oblike $28 \times 28 \times 1$.

---

## 2. Kako CNN in ključni mehanizmi delujejo v ozadju

### 2.1 Gradniki konvolucijske nevronske mreže
- **Konvolucijski sloj (`Conv2D`)**: Po sliki drsi več majhnih učljivih filtrov (jeder) velikosti $3 \times 3$. Vsak filter izračuna skalarni produkt s krajevno okolico pikslov in tako ustvari *mapo značilk* (*feature map*). Zgodnji sloji se običajno naučijo prepoznavati osnovne robove in naklone črt, globlji sloji pa kombinacije teh robov (npr. prave kote kvadrata, ostre vrhove trikotnika ali zvezno ukrivljenost kroga). Ker si vsi deli slike delijo enake uteži filtra, ima CNN veliko manj parametrov kot polno povezana mreža in je bistveno bolj odporen na manjše premike lika.
- **Robno polnjenje (`padding="same"`)**: Brez polnjenja bi jedro $3 \times 3$ na vsakem koraku zmanjšalo višino in širino slike za 2 piksla. Nastavitev `"same"` okoli slike doda en piksel obrobe z ničlami, zato izhodna mapa značilk ohrani enako prostorsko dimenzijo (npr. $28 \times 28$ oziroma $14 \times 14$), robne značilke pa se ne izgubijo prezgodaj.
- **Aktivacijska funkcija `ReLU`**: $f(z) = \max(0, z)$ v mrežo vnese nelinearnost. Brez nelinearne aktivacije bi bilo zaporedje konvolucij in linearnih slojev matematično enakovredno eni sami linearni transformaciji. ReLU je računsko hiter in pri pozitivnih vrednostih ohranja stabilne gradiente.
- **Združevalni sloj (`MaxPooling2D((2, 2))`)**: Na vsakem neprekrivajočem se okencu velikosti $2 \times 2$ izbere največjo vrednost. S tem prostorsko dimenzijo prepolovi ($28 \times 28 \rightarrow 14 \times 14 \rightarrow 7 \times 7$), zmanjša število parametrov v nadaljnjih slojih, pospeši računanje in ustvari krajevno invariantnost na premike za 1–2 piksla.
- **Sploščitveni sloj (`Flatten`)**: Končno tridimenzionalno mapo značilk (npr. $7 \times 7 \times 16 = 784$ ali $3 \times 3 \times 64 = 576$) pretvori v enodimenzionalni vektor, ki služi kot vhod v klasični polno povezani klasifikator.
- **Polno povezani sloj (`Dense`) in `Softmax`**: Skriti sloj `Dense` z aktivacijo `ReLU` združi izluščene prostorske značilke v višjenivojske kombinacije. Izhodni sloj `Dense(3)` z aktivacijo **Softmax** pretvori surove izhode (*logits*) v tri nenegativne vrednosti z vsoto $1,0$, torej v verjetnostno porazdelitev po razredih (`circle`, `square`, `triangle`).

### 2.2 Učenje, funkcija izgube in optimizacija
- **Normalizacija na $[0, 1]$**: Izvorne vrednosti pikslov $[0, 255]$ delimo z `255.0`. Nevronske mreže uporabljajo inicializacijo uteži (npr. Glorot/Xavier), ki pričakuje vhode majhnega reda velikosti; veliki vhodi do 255 bi povzročili prevelike gradiente in nestabilno učenje.
- **`sparse_categorical_crossentropy`**: Za večrazredno klasifikacijo z medsebojno izključujočimi se razredi uporabimo kategorično križno entropijo. Različica `sparse_` omogoča neposredno uporabo celoštevilskih oznak (`0`, `1`, `2`) brez predhodne pretvorbe v *one-hot* vektorje.
- **Optimizator `Adam`**: Prilagodljivi gradientni algoritem, ki za vsako utež vzdržuje drseče povprečje prvega momenta (smer gibanja) in drugega momenta (velikost gradientov). Privzeta hitrost učenja (`0.001`) zanesljivo deluje pri enostavnejših arhitekturah, pri globlji mreži z augmentacijo in manjšim paketkom pa nekoliko višja stopnja (`0.002`) pospeši prehod iz začetnega platoja.

### 2.3 Učinkovit podatkovni cevovod (`tf.data.Dataset`)
Namesto da vse slike ročno kopiramo v pomnilniško polje pred začetkom učenja, `tf.data.Dataset` zgradi pretočni cevovod:
- `.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)` vzporedno prebere datoteke PNG, jih dekodira kot enokanalne sivinske slike (`channels=1`), zagotovi statično obliko `(28, 28, 1)` in vrednosti deli z `255.0`.
- `.shuffle(..., reshuffle_each_iteration=True)` pri učni množici pred vsako epoho na novo premeša vrstni red primerov, da se model ne navadi na zaporedje razredov. Validacijska in testna množica se ne mešata.
- `.batch(batch_size)` združi slike v paketke za učinkovito vektorsko računanje na CPE/GPE.
- `.prefetch(tf.data.AUTOTUNE)` pripravlja naslednji paketek podatkov v ozadju, medtem ko model obdeluje trenutnega.

### 2.4 Stratificirana prečna validacija in ločitev vlog množic
- **5-kratna stratificirana prečna validacija (`StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)`)**: Ker imamo le 300 slik, bi ena sama delitev (npr. 60 testnih slik) imela veliko naključno varianco. S 5-kratno stratificirano delitvijo vsak izmed 5 rezov vsebuje natanko 60 testnih slik (po 20 iz vsakega razreda) in 240 preostalih slik, na koncu pa ocenimo povprečje in standardni odklon čez vseh 300 slik.
- **Tri ločene množice v vsakem rezu**: Iz 240 ne-testnih slik znotraj vsakega reza s `train_test_split(..., test_size=0.2, stratify=...)` ločimo **192 učnih slik** (64 na razred) in **48 validacijskih slik** (16 na razred).
  - **Učna množica (192 slik)**: služi izračunu gradientov in posodabljanju uteži (ter edina prejema podatkovno augmentacijo).
  - **Validacijska množica (48 slik)**: služi sprotnemu spremljanju učenja (`val_loss`, `val_accuracy`) in zgodnjemu ustavljanju (`EarlyStopping`).
  - **Testna množica (60 slik)**: ostane strogo ločena do konca reza in služi izključno za nepristranski izračun končnih metrik.
- **Sveža inicializacija**: V vsakem prečnem rezu se kliče `keras.backend.clear_session()` in ustvari popolnoma nov model, da ne pride do prenosa naučenih uteži in uhajanja podatkov med rezi.

### 2.5 Podatkovna augmentacija, regularizacija in zgodnje ustavljanje
- **Podatkovna augmentacija (`RandomFlip`, `RandomTranslation`)**: Z vsako epoho naključno zrcalimo (`horizontal_and_vertical`) in rahlo premikamo (`0.06`, tj. do ~1,7 piksla) učne slike. Ker gre za geometrijske oblike, zrcaljenje in majhen premik kroga, kvadrata ali trikotnika nikoli ne spremenita pravega razreda lika.
- **Ključna nastavitev `fill_mode='constant', fill_value=1.0`**: Slike v zbirki imajo **belo ozadje** (vrednost `1.0` po normalizaciji) in temne robove likov. Če bi pri premiku uporabili privzeto polnjenje z ničlami (`0.0`), bi se ob robovih slike pojavili črni trakovi, ki bi jih konvolucijski filtri zamenjali za umetne robove lika. Z `fill_value=1.0` novi robni piksli ostanejo beli.
- **Vgrajenost augmentacije v model**: Kerasovi sloji za augmentacijo so aktivni izključno med učenjem (`training=True`) in se samodejno izklopijo pri vrednotenju validacijske in testne množice (`training=False`).
- **Regularizacija `Dropout(0.25)` in `L2(1e-4)`**: `Dropout` med učenjem naključno izklopi 25 % aktivacij pred polno povezanim slojem, kar prepreči soodvisnost posameznih nevronov. Kazen `L2` v funkcijo izgube doda $\lambda \sum w^2$ in s tem omejuje prevelike uteži ter gladi odločitvene meje.
- **`EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`**: Spremlja validacijsko izgubo; če se ta ne izboljša 10 zaporednih epoh, učenje prekine in povrne uteži iz epohe z najnižjo vrednostjo `val_loss`.

---

## 3. Pregled postopka korak za korakom in utemeljitev izbir

### Korak 1: Priprava podatkov in pregled razredov
- Iz map `shapes/circles`, `shapes/squares` in `shapes/triangles` zgradimo preglednico `DataFrame` s stolpcema `filepath` in `label` (300 vrstic, po 100 na razred).
- Besedilne oznake preslikamo v števila `{'circle': 0, 'square': 1, 'triangle': 2}` in zgradimo cevovod `tf.data.Dataset`, ki vsako sliko pretvori v tenzor oblike `(28, 28, 1)` z vrednostmi tipa `float32` v intervalu `[0.0, 1.0]` (dejanski izmerjen razpon na prvem paketku: `0.059 – 1.0`).
- **Zakaj `tf.image.resize(img, [28, 28])`?** Čeprav so slike že dimenzij $28 \times 28$, funkcija TensorFlow grafu zagotovi znano statično obliko `(28, 28, 1)` in pretvorbo v `float32`, kar je potrebno za povezavo z vhodno plastjo `Input(shape=(28, 28, 1))`.

### Korak 2: Izbira postopkov augmentacije
- Izbrani sta dve transformaciji: `RandomFlip('horizontal_and_vertical')` in `RandomTranslation(0.06, 0.06, fill_mode='constant', fill_value=1.0)`.
- **Zakaj takšen izbor?** Zrcaljenje ohranja ostre robove črt in poveča raznolikost usmerjenosti (npr. trikotnik obrnjen navzgor ali navzdol), zmerni premik do 6 % (~1,7 piksla) pa model naredi odporen na necentriranost ročno narisanih likov, brez da bi lik odrezal ob robu slike. Večje rotacije na mreži velikosti $28 \times 28$ zaradi bilinearne interpolacije zameglijo tanke enopikselne črte, kar pri plitvih mrežah oteži učenje znotraj 50 epoh.

### Korak 3: Referenčna in prilagojena arhitektura CNN

| Lastnost | Referenčni CNN (A, B) | Prilagojeni CNN (C) |
|---|---|---|
| **Konvolucijski bloki** | 2 sloja: `16 -> 16` filtrov (`3x3`, `same`) | 3 bloki: `16 -> 32 -> 64` filtrov (`3x3`, `same`) |
| **Združevanje** | 2 × `MaxPooling2D(2x2)` (`28 -> 14 -> 7`) | 3 × `MaxPooling2D(2x2)` (`28 -> 14 -> 7 -> 3`) |
| **Vektor po `Flatten`** | `7 × 7 × 16 = 784` | `3 × 3 × 64 = 576` |
| **Polno povezani sloj** | `Dense(64, relu)` | `Dropout(0.25)` + `Dense(32, relu, L2=1e-4)` |
| **Učljivi parametri (skupaj)** | **52.915** | **41.859** (~20,9 % manj) |
| **Parametri v konvoluciji** | 2.480 (4,7 % vseh) | 23.296 (55,7 % vseh) |
| **Parametri v `Dense` glavi** | 50.435 (95,3 % vseh) | 18.563 (44,3 % vseh) |
| **Učni hiperparametri** | `Adam(lr=0.001)`, `batch_size=32` | `Adam(lr=0.002)`, `batch_size=16` |

- **Utemeljitev sprememb pri modelu C**:
  1. **Prerazporeditev kapacitete iz `Dense` v `Conv2D`**: Referenčni model porabi skoraj 95 % vseh uteži na prehodu `Flatten(784) -> Dense(64)` (50.240 parametrov), medtem ko ima konvolucijski del le 2.480 parametrov. Dodaten konvolucijski blok s progresivnim številom filtrov (`16 -> 32 -> 64`) in tretje združevanje `MaxPooling2D` zmanjšata prostor pred `Flatten` na 576 elementov ter s tem znižata skupno število parametrov na **41.859**, hkrati pa močno povečata zmožnost zajema kompleksnejših geometrijskih vzorcev.
  2. **Regularizacija (`Dropout(0.25)` in `L2(1e-4)`)**: Zmanjšujeta tveganje preprileganja pri večji globini mreže na zbirki 192 učnih slik.
  3. **Manjši paketek (`batch_size=16`) in `learning_rate=0.002`**: Na 192 učnih slikah paketek velikosti 32 izvede le 6 posodobitev uteži na epoho, kar je ob podatkovni augmentaciji in regularizaciji prepočasi za 50 epoh. Paketek velikosti 16 podvoji število korakov na 12 na epoho, nekoliko višja učna stopnja `0.002` pa omogoči hiter zagon učenja brez nestabilnosti.

### Korak 4: Primerjava treh konfiguracij (A, B, C)
- **A**: Referenčni CNN brez augmentacije (`batch_size=32`, `lr=0.001`).
- **B**: Referenčni CNN z augmentacijo (`batch_size=32`, `lr=0.001`).
- **C**: Prilagojeni CNN z augmentacijo in regularizacijo (`batch_size=16`, `lr=0.002`).

---

## 4. Povzetek dejanskih rezultatov

### Primerjalna tabela (5-kratna stratificirana prečna validacija)

| Konfiguracija | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|:---:|:---:|:---:|:---:|
| **A: Referenčni CNN** | `0.827 ± 0.068` | `0.832 ± 0.065` | `0.827 ± 0.068` | `0.824 ± 0.071` |
| **B: Referenčni CNN + augmentacija** | `0.817 ± 0.089` | `0.823 ± 0.090` | `0.817 ± 0.089` | `0.814 ± 0.092` |
| **C: Prilagojeni CNN** | **`0.920 ± 0.062`** | **`0.924 ± 0.061`** | **`0.920 ± 0.062`** | **`0.920 ± 0.061`** |

### Ključne ugotovitve
- **Primerjava s klasičnimi pristopi**: Referenčni CNN že brez augmentacije (82,7 %) bistveno presega modele nad sploščenimi vektorji slik (odločitveno drevo ~43,3 %, KNN ~53,3 %, MLP brez prostorskih filtrov ~47,7 %), saj konvolucijska jedra neposredno zaznavajo robove, kote in ukrivljenost ne glede na položaj lika.
- **Vpliv augmentacije na plitvi referenčni model (A vs. B)**: V 50 epohah ima konfiguracija B nekoliko nižje povprečje (0,817 proti 0,827) in večji standardni odklon (0,089), čeprav na 5. rezu doseže 0,950. Iz učnih krivulj 1. reza je razvidno, da pri modelu A po 25. epohi učna točnost (~0,93) že prehiteva validacijsko (~0,81), medtem ko pri modelu B augmentacija prepreči preprileganje (učna točnost ~0,78 pod validacijsko ~0,83), vendar plitva mreža s 6 posodobitvami na epoho v 50 epohah še ne konvergira do konca.
- **Prednost prilagojenega modela (C)**: Konfiguracija C kljub ~21 % manjšem številu vseh parametrov zanesljivo preseže oba referenčna modela z **92,0 % točnosti** (na 3. in 4. rezu kar **96,7 %**) in najnižjim standardnim odklonom (`0.062`). Učna in validacijska izguba se po začetni fazi strmo in usklajeno spustita vse do ~0,14 oziroma ~0,07 v 50. epohi.
- **Analiza napak (matrika zmede na 1. rezu, 60 testnih slik)**:
  - Model **A** (46/60 pravilnih) najpogosteje zamenjuje kroge (`circle`: 12/20 pravilnih, po 4 zamenjave s kvadratom in trikotnikom).
  - Model **B** (45/60 pravilnih) dosega po 17/20 pri kvadratih in trikotnikih, 8 krogov pa napačno uvrsti med trikotnike.
  - Model **C** (56/60 pravilnih) brez ene same napake pravilno razvrsti vseh **20 kvadratov** in vseh **20 trikotnikov**, edine 4 napake pa se pojavijo pri ročno narisanih krogih z nekoliko bolj oglatimi obrisi (2 kot kvadrat, 2 kot trikotnik).

---

## 5. Ključna vprašanja in odgovori za predstavitev

**V: Zakaj CNN na teh slikah deluje bistveno bolje kot MLP, KNN ali odločitveno drevo?**
> **O**: Klasični modeli sliko sploščijo v vektor 784 neodvisnih značilk. Če se krog ali kvadrat na sliki premakne za 2–3 piksele, se vrednosti posameznih vhodnih stolpcev popolnoma spremenijo. CNN uporablja učljive konvolucijske filtre $3 \times 3$ in združevanje `MaxPooling2D`, ki zaznavajo lokalne oblike (robove, prave/ostre kote, ukrivljenost) neodvisno od natančne lege lika na sliki.

**V: Zakaj smo v prilagojenem modelu zmanjšali skupno število parametrov (s 52.915 na 41.859) in vseeno dosegli za skoraj 10 odstotnih točk višjo točnost?**
> **O**: V referenčnem modelu je 95 % parametrov v polno povezani plasti `Flatten(784) -> Dense(64)`, kar spodbuja pomnjenje položajev pikslov namesto učenja oblike. V prilagojenem modelu smo dodali tretji konvolucijski blok (`16 -> 32 -> 64`) in tretje združevanje `MaxPooling2D`, kar je izluščilo bogatejše hierarhične značilke v konvolucijskem delu (23.296 proti 2.480 parametrom) in zmanjšalo število parametrov v polno povezani glavi.

**V: Zakaj je bilo pri augmentaciji nujno nastaviti `fill_mode='constant', fill_value=1.0`?**
> **O**: Ker so slike v zbirki narisane s temnimi črtami na belem ozadju (vrednost `1.0` po normalizaciji na `[0, 1]`). Privzeto polnjenje s `0.0` bi pri premiku slike ustvarilo črne robove, ki bi za mrežo predstavljali umetne črte in pokvarili učenje.

**V: Zakaj augmentacije ne smemo uporabljati na validacijski in testni množici?**
> **O**: Validacijska in testna množica predstavljata resnične, nepredelane vhodne podatke za merjenje dejanske sposobnosti posploševanja. Kerasovi sloji za augmentacijo so zgrajeni tako, da transformacije izvajajo izključno med klicem učenja (`training=True`) in med validacijo ter napovedovanjem (`training=False`) slike prepustijo nespremenjene.

**V: Kakšna je vloga validacijske množice v primerjavi s testno množico znotraj posameznega prečnega reza?**
> **O**: Validacijska množica (48 slik) neposredno vpliva na proces učenja, saj na njej spremljamo krivulji `val_loss` in `val_accuracy` ter sprožamo zgodnje ustavljanje (`EarlyStopping`) z obnovitvijo najboljših uteži. Testna množica (60 slik) v procesu učenja ali ustavljanja sploh ne sodeluje, zato daje nepristransko oceno končne uspešnosti.

**V: Zakaj smo v prilagojenem modelu zmanjšali `batch_size` s 32 na 16 in povečali `learning_rate` na `0.002`?**
> **O**: Pri 192 učnih slikah paketek velikosti 32 pomeni samo 6 gradientnih korakov na epoho (300 korakov v 50 epohah), kar ob naključni augmentaciji in regularizaciji `Dropout` ni dovolj za popolno konvergenco. Zmanjšanje paketka na 16 podvoji število posodobitev na epoho (600 korakov), nekoliko višja učna stopnja `0.002` pa pospeši učenje globlje mreže brez izgube stabilnosti.
