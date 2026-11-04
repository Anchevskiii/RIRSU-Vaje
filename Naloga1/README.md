# Naloga 1 – Regresijska analiza nad tabelaričnimi podatki

V tej nalogi je cilj zgraditi regresijski model za napovedovanje števila izposojenih mestnih koles (`rented_bike_count`) na podlagi vremenskih in koledarskih dejavnikov.

## Podatki (`bike_data.csv`)
Podatkovna zbirka vsebuje 8.760 urnih meritev z naslednjimi značilkami:
- **Koledarski podatki**: `date`, `hour`, `seasons`, `holiday`, `work_hours`.
- **Vremenske spremenljivke**: `temperature`, `humidity`, `wind_speed`, `visibility`, `dew_point_temperature`, `solar_radiation`, `rainfall`, `snowfall`.
- **Ciljna spremenljivka**: `rented_bike_count` (število izposojenih koles na določeno uro).

## Potek delovanja (Pipeline)

1. **Pregled podatkov in manjkajoče vrednosti**:
   - Ugotovljeno je, da ima samo stolpec `dew_point_temperature` 86 manjkajočih vrednosti (`NaN`).
2. **Koledarska predobdelava**:
   - Stolpec `date` (format `dd/mm/yyyy`) se razdeli na tri numerične stolpce: `day`, `month`, `year`. Izvirni stolpec `date` se nato zavrže.
3. **Delitev podatkov (Train / Test split)**:
   - Podatki se razdelijo na učno (70 %) in testno (30 %) množico s fiksnim semenom `random_state=1234`.
4. **Zapolnjevanje manjkajočih vrednosti in kodiranje**:
   - Manjkajoče vrednosti v `dew_point_temperature` se zapolnijo s **povprečjem učne množice** (preprečevanje uhajanja podatkov).
   - Kategorične spremenljivke (`seasons`, `holiday`, `work_hours`) se pretvorijo v številske z One-Hot kodiranjem (`pd.get_dummies(..., drop_first=True)`).
5. **Učenje osnovnega modela**:
   - Nauči se model **navadne linearne regresije** (`LinearRegression()`).
6. **Vrednotenje modela**:
   - Model se ovrednoti na testni množici z metrikami:
     - **MAE (Mean Absolute Error)**: Povprečna absolutna napaka (~330 koles).
     - **MSE (Mean Squared Error)**: Povprečna kvadratna napaka (~193.318).
     - **$R^2$ (Determinacijski koeficient)**: ~0,554 (model pojasni ~55 % variance).
     - **EVS (Explained Variance Score)**: ~0,554 (enako kot $R^2$, kar pomeni odsotnost sistematičnega premika napake).
7. **Dodatni del – Primerjava z algoritmom KNN**:
   - Preizkusi se algoritem K-najbližjih sosedov za regresijo (`KNeighborsRegressor`) s parametri $k = 3$, $k = 5$ in $k = 10$.
   - **Ugotovitev**: Model s $k=5$ doseže nekoliko nižji MAE (~306) kot linearna regresija, vendar je linearna regresija hitrejša in stabilnejša pri ekstrapolaciji.
