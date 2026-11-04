# Naloga 2 – Klasifikacija geometrijskih slik

Cilj te naloge je razvrščanje sivinskih slik preprostih geometrijskih likov v tri razrede: **krog (`circle`)**, **kvadrat (`square`)** in **trikotnik (`triangle`)**.

## Podatki (`shapes/`)
Podatkovna zbirka vsebuje skupaj 300 slik formata PNG:
- `shapes/circles/`: 100 slik krogov
- `shapes/squares/`: 100 slik kvadratov
- `shapes/triangles/`: 100 slik trikotnikov
- Slike so velikosti $28 \times 28$ slikovnih pik (skupaj 784 slikovnih pik na sliko).

## Potek delovanja (Pipeline)

1. **Nalaganje in pretvorba slik**:
   - Vsaka slika se odpre s knjižnico Pillow in pretvori v sivinsko obliko (`img.convert('L')`).
   - Dvodimenzionalna matrika pikslov se splošči v enodimenzionalni vektor dolžine 784 z vrednostmi od 0 do 255 (`np.array(img).flatten()`).
2. **Stratificirana delitev podatkov**:
   - Množica se razdeli na učno (70 % = 210 slik) in testno (30 % = 90 slik).
   - Uporabljen je `stratify=y`, ki zagotavlja natanko po 30 slik vsakega razreda v testni množici (`random_state=4321`).
3. **Učenje modela odločitvenega drevesa**:
   - Zgrajeno je odločitveno drevo (`DecisionTreeClassifier(random_state=1234)`).
   - Drevo izvaja pravokotne razcepe prostora glede na svetlost posameznih točk (pikslov).
4. **Vrednotenje in matrika zmede**:
   - Izračunane so metrike na testni množici:
     - **Točnost (Accuracy)**: ~43,3 % (bolje od naključnega 33,3 %, vendar skromno).
     - **Utežena F1-vrednost**: ~0,435.
     - **Utežena preciznost**: ~0,441.
     - **Utežen priklic**: ~0,433.
   - **Matrika zmede (*Confusion Matrix*)**: Razkrije, da drevo najpogosteje zamenjuje kvadrate in kroge, medtem ko trikotnike prepoznava najbolj zanesljivo (zaradi specifičnih vogalnih točk).
5. **Dodatni del – Primerjava z algoritmom KNN**:
   - Natrenirani so modeli K-najbližjih sosedov (`KNeighborsClassifier`) s $k \in \{3, 5, 10\}$.
   - **Ključna ugotovitev**: Vsi modeli KNN dosežejo precej boljšo točnost (**50,0 % – 53,3 %**) kot odločitveno drevo (43,3 %).
   - **Razlaga**: KNN meri evklidsko razdaljo med celotnimi vektorji slik (globalna podobnost celotne oblike), medtem ko posamezno odločitveno drevo gleda le izolirane piksle, kar pri necentriranih likih hitro privede do prekomernega prilagajanja (*overfitting*).
