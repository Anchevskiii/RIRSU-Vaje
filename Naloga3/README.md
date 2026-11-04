# Naloga 3 – Nevronske mreže (MLP), predobdelava podatkov in hiperparametri

Ta dokument predstavlja celovit in poglobljen vodnik po 3. nalogi (tako **obvezni** kot **dodatni del** – skupaj 4 točke) pri predmetu RIRSU.
Celotna programska koda z vsemi vnaprej izvedenimi izpisi, tabelami in grafi se nahaja v interaktivnem zvezku [naloga3.ipynb](file:///C:/Sola/FERI/Magistrski/1.%20Letnik/RIRSU/Vaje/Naloga3/naloga3.ipynb).

---

## 1. Kako nevronske mreže (MLP) delujejo pod pokrovom?

Večplastni perceptron (**Multi-Layer Perceptron – MLP**) je osnovna arhitektura naprej usmerjenih (*feedforward*) nevronskih mrež. 

### 1.1 Zgradba nevrona in plasti
- **Vhodna plast (Input layer)**: Sprejme vhodni vektor značilk $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ (npr. vremenske spremenljivke ali 784 pikslov slike).
- **Skrite plasti (Hidden layers)**: Vsak nevron v skriti plasti izračuna uteženo vsoto vseh vhodov, prišteje prosti člen (*bias* $b$) ter rezultat pošlje skozi nelinearno aktivacijsko funkcijo $f$:
  $$z = \sum_{i=1}^d w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
  $$a = f(z)$$
- **Aktivacijska funkcija**: Brez nje bi bila nevronska mreža zgolj zaporedje linearnih transformacij (tj. navadna linearna regresija). Nelinearnost omogoča modeliranje poljubno zapletenih funkcij:
  - **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$. Je računska zelo hitra, ne trpi za izginjajočim gradientom pri pozitivnih vrednostih, vendar lahko negativne vrednosti povsem "zamrzne" (*dead neurons*).
  - **Tanh (Hiperbolični tangens)**: $f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Vrednosti stisne v simetričen interval $(-1, 1)$, kar daje gladke odvode in je primerno za zvezne regresijske napovedi.
- **Izhodna plast (Output layer)**:
  - Za **regresijo**: En nevron z linearno aktivacijo (napove zvezno število koles).
  - Za **klasifikacijo**: Trije nevroni z verjetnostno funkcijo *Softmax*, ki vrnejo verjetnosti za vsakega od 3 razredov (krog, kvadrat, trikotnik).

### 1.2 Učenje z vzvratnim širjenjem napake (*Backpropagation*) in optimizator Adam
1. **Prehod naprej (*Forward pass*)**: Vhodni podatki potujejo skozi mrežo do izhoda, kjer model naredi napoved $\hat{y}$.
2. **Izračun funkcije izgube (*Loss function*)**:
   - Regresija: Srednja kvadratna napaka (MSE Loss) $= \frac{1}{n} \sum (y - \hat{y})^2$.
   - Klasifikacija: Križna entropija (Cross-Entropy / Log Loss) $= -\sum y \log(\hat{y})$.
3. **Prehod nazaj (*Backward pass*)**: Z uporabo verižnega pravila odvajanja (*chain rule*) se izračunajo parcialni odvodi funkcije izgube glede na vsako posamezno utež $\frac{\partial \mathcal{L}}{\partial w}$.
4. **Posodobitev uteži (Adam optimizator)**: Uteži se popravijo v nasprotni smeri gradienta z uporabo prilagodljive hitrosti učenja (*adaptive learning rate*) in gibalne količine (*momentum*).

### 1.3 Zakaj je standardizacija/normalizacija ključna?
Nevronske mreže so izjemno občutljive na merilo vhodnih podatkov:
- Če ima en vhod vrednost npr. $2000$ (vidljivost), drugi pa $0.2$ (količina padavin), bodo gradienti za vidljivost ogromni, za padavine pa zanemarljivi.
- Velike številke povzročijo, da nevroni takoj vstopijo v območje nasičenja ali pa uteži eksplodirajo, kar privede do nedosežene konvergence (`ConvergenceWarning`).
- Standardizacija (`StandardScaler`: povprečje 0, varianca 1) ali normalizacija ($[0, 1]$) postavi vse vhode na enakovredno raven.

---

## 2. Podroben pregled zvezka `naloga3.ipynb` po blokih

---

### BLOK 1: Uvoz knjižnic in priprava okolja
- **Koda**: Uvoz orodij iz `sklearn` (`MLPRegressor`, `MLPClassifier`, `KFold`, `StratifiedKFold`, `PowerTransformer`, `StandardScaler`, `mutual_info_regression`), `pandas`, `numpy`, `matplotlib` in `PIL`.
- **Namen**: Priprava celotnega cevovoda in fiksiranje naključnega semena (`random_state=1234`) za popolno ponovljivost vseh eksperimentov.

---

### BLOK 2: Regresijski problem – Priprava podatkov (Obvezni del)
Cilj je napovedati urno število izposojenih koles `rented_bike_count`.

#### Različica A – Osnovna predobdelava
- Datum razbijemo na `day`, `month`, `year`.
- Manjkajoče vrednosti `dew_point_temperature` (86 vrstic) zapolnimo s preprostim **povprečjem**.
- Kategorične stolpce pretvorimo z *One-Hot* kodiranjem.
- **Pomembno**: Podatkov **ne skaliramo** – namerno ohranimo surove vrednosti, da preverimo obnašanje nevronske mreže na neskaliranih podatkih.

#### Različica B – Napredna predobdelava
1. **Inženiring 3 novih značilk**:
   - `is_rush_hour`: Ura je med 7–9h ali 17–19h ob delavnikih (čas prihodov v službo in odhodov domov).
   - `temp_humidity_comfort`: Značilka toplotnega ugodja ($T \times (1 - \text{humidity}/100)$). Združuje vpliv toplote in soparnosti.
   - `bad_weather`: Binarna oznaka padavin (`rainfall > 0` ali `snowfall > 0`).
2. **Imputacija z linearno regresijo**:
   - Model `LinearRegression()` se natrenira na ostalih vremenskih spremenljivkah in zelo natančno napove manjkajoče vrednosti temperature rosišča.
3. **Yeo-Johnson transformacija (`PowerTransformer`)**:
   - Spremenljivki `solar_radiation` in `wind_speed` imata močno asimetrično porazdelitev (ogromno vrednosti je 0, le nekaj je zelo visokih). Transformacija ju preoblikuje v približno normalno (Gaussovo) porazdelitev.
4. **Standardizacija (`StandardScaler`)**:
   - Vse številske spremenljivke se preslikajo na $\mu = 0$ in $\sigma = 1$.
5. **Izbira značilk z medsebojno informacijo (`mutual_info_regression`)**:
   - Nelinearna ocena pomembnosti značilk. Značilka `temp_humidity_comfort` se izkaže za najpomembnejšo v celotnem naboru ($MI \approx 0.445$). Izbere se najboljših 10 značilk za zmanjšanje šuma.

---

### BLOK 3: 5-kratna prečna validacija regresije in krivulje izgube
- **Metodologija preprečevanja uhajanja podatkov (*Zero Data Leakage*)**:
  - Podatke s `KFold(n_splits=5, shuffle=True, random_state=1234)` razdelimo na 5 rezov.
  - VSAKA transformacija (računanje povprečja, treniranje linearnega regresorja za imputacijo, določitev parametrov Yeo-Johnson, računanje $\mu$ in $\sigma$ za skaliranje ter izbira top 10 značilk) se izvede **izključno na 80% učnih podatkih** posameznega reza!
  - Testni podatki (20%) se zgolj transformirajo z že naučenimi transformatorji.
- **Krivulja izgube (*Loss curve*)**:
  - Pri Različici A se optimizator ustavi pri meji 200 iteracij z opozorilom `ConvergenceWarning`, izguba pa ostane visoka.
  - Pri Različici B funkcija izgube gladko in strmo pade proti bistveno nižji vrednosti.
- **Rezultati regresije (5-fold CV)**:
  - **Različica A**: $R^2 = 0.546 \pm 0.017$, MAE $= 307.7 \pm 7.7$
  - **Različica B**: **$R^2 = 0.678 \pm 0.020$**, **MAE $= 250.7 \pm 11.2$** (zmanjšanje napake za skoraj 20%!).

---

### BLOK 4: Klasifikacija geometrijskih oblik (Obvezni del)
Cilj je razvrstiti 300 slik velikosti $28 \times 28$ v tri razrede: krog, kvadrat in trikotnik.
- **Različica A (Osnovna)**: Sivinske slike sploščene v vektor 784 številk v izvornem območju **[0, 255]**.
- **Različica B (Napredna)**: Sivinske slike z vrednostmi **normaliziranimi na interval [0.0, 1.0]** ($X / 255.0$).
- **Vrednotenje**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)` zagotavlja enako zastopanost vseh treh likov v vsakem rezu.
- **Rezultati**:
  - Različica A: Točnost $= 38.0 \pm 2.9\%$, Weighted F1 $= 0.302$
  - Različica B: **Točnost $= 47.7 \pm 4.8\%$**, **Weighted F1 $= 0.438$**
  - *Razlaga*: Ker inicializacija uteži v nevronski mreži (Xavier/Glorot) predpostavlja vhode majhnih vrednosti, vhodni signali med 0 in 255 povzročijo prenasičenje nevronov. Normalizacija na $[0, 1]$ omogoči stabilno učenje.

---

### BLOK 5: Dodatni del – Napredna analiza regresije (Ablation Study)
Za podrobno razumevanje vpliva vsakega posameznega koraka primerjamo 5 različic predobdelave na identičnih prečnih rezih:

1. **Dodatni inženiring značilk (skupaj 5 značilk)**:
   - Značilka 4: `is_weekend` (`dayofweek >= 5`) – sobota in nedelja imata popolnoma drugačen profil izposoje (rekreativna uporaba čez dan namesto jutranje konice).
   - Značilka 5: `wind_chill` – formula občutene temperature zaradi vetra ($13.12 + 0.6215 T - 11.37 V^{0.16} + 0.3965 T V^{0.16}$).
2. **Alternativna imputacija (`KNNImputer`)**:
   - Manjkajoče rosišče se zapolni z iskanjem 5 vremensko najbolj podobnih ur v učnem naboru.
3. **Rezultati primerjave 5 različic**:

| Različica predobdelave | MAE | MSE | $R^2$ | EVS |
|---|:---:|:---:|:---:|:---:|
| **A (Osnovna - povprečje, brez skaliranja)** | $307.7 \pm 7.7$ | $188807 \pm 9304$ | $0.546 \pm 0.017$ | $0.548 \pm 0.018$ |
| **B (Napredna - LinReg, 3 zn., top 10)** | $250.7 \pm 11.2$ | $124357 \pm 10183$ | $0.678 \pm 0.020$ | $0.679 \pm 0.020$ |
| **C (Napredna - Vse značilke)** | **$212.9 \pm 10.4$** | **$87262 \pm 6288$** | **$0.774 \pm 0.013$** | **$0.775 \pm 0.013$** |
| **D (Napredna - KNNImputer)** | $250.6 \pm 11.4$ | $124239 \pm 10142$ | $0.678 \pm 0.020$ | $0.679 \pm 0.020$ |
| **E (Razširjena - 5 značilk)** | $255.4 \pm 12.3$ | $130282 \pm 9642$ | $0.662 \pm 0.020$ | $0.663 \pm 0.019$ |

- **Ključna ugotovitev (Ablation study)**:
  - `KNNImputer` in `LinearRegression` data praktično enak rezultat (ker je manjkalo le 86 vrstic).
  - **Različica C (uporaba vseh značilk brez filtriranja) doseže izjemen $R^2 = 0.774$**! Razlog: Nevronska mreža ima dovolj parametrov, da sama uteži in združi vse nelinearne interakcije, če so podatki le pravilno skalirani. Omejitev na top 10 značilk je modelu odvzela nekaj koristnih informacij.

---

### BLOK 6: Dodatni del – Zaznavanje robov s Sobelovim filtrom
- **Kako deluje Sobelov filter**:
  Vsako $28 \times 28$ sliko konvoluiramo z dvema $3 \times 3$ jedroma:
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
  Nato izračunamo magnitudo gradienta $G = \sqrt{G_x^2 + G_y^2}$ ter vrednosti normaliziramo na $[0, 1]$.
- **Vizualizacija**: V zvezku je prikazana matrika primerjalnih slik (izvirna sivinska slika vs. izluščeni robovi za krog, kvadrat in trikotnik).
- **Rezultati klasifikacije (5-fold CV)**:
  - **Različica A (Surove 0–255)**: Točnost $= 41.0 \pm 6.2\%$
  - **Različica B (Normalizirane 0–1)**: Točnost $= 47.7 \pm 4.8\%$
  - **Različica C (Sobel robovi 0–1)**: **Točnost $= 55.0 \pm 5.3\%$**, **Weighted F1 $= 0.518 \pm 0.053$**
- **Zakaj je Sobel tako učinkovit?**
  Običajen MLP obravnava piksle kot neodvisne spremenljivke brez prostorske zavesti (nima konvolucijskih jeder). Sobelov filter izpostavi kote, oglišča in ukrivljenost ter izbriše enolično ozadje, kar modelu omogoči neposredno razlikovanje med ostrimi koti trikotnika, pravimi koti kvadrata in zvezno ukrivljenostjo kroga.

---

### BLOK 7: Dodatni del – Uglaševanje arhitektur in hiperparametrov
Preizkusili smo 4 premišljene konfiguracije mrež za vsak problem pri fiksni predobdelavi:

#### Regresijski problem (`MLPRegressor`):
1. **Config 0 (Privzeta)**: `(100,)`, `relu`, $\alpha=0.0001$, `lr=0.001`, `max_iter=200` $\rightarrow R^2 = 0.478 \pm 0.029$
2. **Config 1 (Globlja 2-plastna)**: `(128, 64)`, `relu`, $\alpha=0.0001$, `max_iter=300` $\rightarrow R^2 = 0.559 \pm 0.008$
   *(Utemeljitev: 2 plasti omogočata učenje nelinearnih hierarhij, npr. čas + vreme).*
3. **Config 2 (Regularizirana)**: `(100, 50)`, `relu`, $\alpha=0.01$, `max_iter=400` $\rightarrow R^2 = 0.560 \pm 0.007$
   *(Utemeljitev: L2 regularizacija $\alpha=0.01$ zatre nihanje uteži in prepreči prenaučenost).*
4. **Config 3 (Tanh aktivacija)**: `(100,)`, `tanh`, $\alpha=0.001$, `lr=0.005`, `max_iter=400` $\rightarrow \mathbf{R^2 = 0.575 \pm 0.004}$
   *(Utemeljitev: Zvezna in gladka aktivacija tanh z zalogo vrednosti $(-1, 1)$ omogoča stabilno modeliranje zveznih vrednosti).*

#### Klasifikacijski problem (`MLPClassifier`):
1. **Config 0 (Privzeta)**: `(100,)`, `relu`, $\alpha=0.0001$, `max_iter=200` $\rightarrow \text{Točnost} = 47.7 \pm 4.8\%$
2. **Config 1 (Povečana kapaciteta)**: `(256, 128)`, `relu`, $\alpha=0.0001$, `max_iter=300` $\rightarrow \text{Točnost} = 44.3 \pm 6.5\%$
   *(Pojasnilo: Ker imamo le 300 slik, prevelika mreža memorizira posamezne slike – overfitting).*
3. **Config 2 (Visoka L2 regularizacija)**: `(100,)`, `relu`, $\mathbf{\alpha=0.1}$, `max_iter=300` $\rightarrow \mathbf{\text{Točnost} = 49.3 \pm 1.7\%}$
   *(Pojasnilo: Visoka kazen $\alpha=0.1$ prisili mrežo k preprostejšim in robustnejšim odločitvenim mejam, kar prinese najvišjo točnost in najmanjšo varianco!).*
4. **Config 3 (Tanh aktivacija)**: `(128, 64)`, `tanh`, $\alpha=0.001$, `lr=0.0005`, `max_iter=300` $\rightarrow \text{Točnost} = 45.7 \pm 6.2\%$

---

## 3. Pogosta vprašanja in odgovori za zagovor (Q&A Cheat Sheet)

**V: Zakaj je model na neskaliranih podatkih (Različica A) vrnil `ConvergenceWarning`?**
> **O**: Ker so vrednosti nekaterih značilk (npr. vidljivost do 2000) nekaj tisočkrat večje od drugih, so gradienti preveliki, optimizator Adam pa v privzetih 200 epohah ni uspel doseči minimuma funkcije izgube.

**V: Kako ste preprečili uhajanje podatkov (*data leakage*) pri transformacijah in imputaciji?**
> **O**: Vsi parametri (povprečje, uteži regresorja za manjkajoče vrednosti, $\lambda$ parameter pri Yeo-Johnson, povprečje in standardni odklon pri skaliranju ter ocena medsebojne informacije) so bili izračunani **izključno na učnih podatkih posameznega prečnega reza**. Testni del vsakega reza je bil transformiran z že naučenimi objekti.

**V: Zakaj je Sobelov filter dvignil točnost klasifikacije geometrijskih oblik za več kot 7%?**
> **O**: Preprost večplastni perceptron nima vgrajene translacijske invariantnosti niti prostorskih filtrov kot konvolucijske mreže (CNN). Sobelov filter vnaprej izračuna prostorske gradiente ter izpostavi robove in oglišča, kar modelu omogoči neposredno prepoznavo geometrijskih značilnosti likov brez motenj ozadja.

**V: Zakaj se je pri slikah model z večjo kapaciteto `(256, 128)` obnesel slabše od preprostega regulariziranega modela?**
> **O**: Zaradi majhnosti podatkovne zbirke (le 300 slik). Pri velikem številu parametrov mreža hitro zapade v prenaučenost (*overfitting*) – začne si dobesedno pomniti šum v učnih slikah namesto splošnih oblik. Močna regularizacija ($\alpha = 0.1$) pri Config 2 je ta pojav učinkovito preprečila.
