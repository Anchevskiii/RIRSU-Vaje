# uvoz knjiznic in priprava poti do slik ter oznak
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

base_dir = 'shapes'
categories = ['circle', 'square', 'triangle']
folder_map = {'circle': 'circles', 'square': 'squares', 'triangle': 'triangles'}

image_paths = []
labels = []

for cat in categories:
    cat_dir = os.path.join(base_dir, folder_map[cat])
    for fname in sorted(os.listdir(cat_dir)):
        if fname.endswith('.png'):
            image_paths.append(os.path.join(cat_dir, fname))
            labels.append(cat)

# prikaz primerov slik iz vsake kategorije
fig, axes = plt.subplots(1, len(categories), figsize=(9, 3))
for ax, cat in zip(axes, categories):
    idx = labels.index(cat)
    with Image.open(image_paths[idx]) as img:
        ax.imshow(img)
    ax.set_title(cat)
    ax.axis('off')
plt.tight_layout()
plt.show()

# preverjanje dimenzij slik in stevila primerov po razredih
sizes = set()
for path in image_paths:
    with Image.open(path) as img:
        sizes.add(img.size)

print('Dimenzije slik:', sizes)
for cat in categories:
    print(f'{cat}: {labels.count(cat)}')

# priprava polja vhodnih podatkov X in oznak y
X = []
for path in image_paths:
    with Image.open(path) as img:
        gray = img.convert('L')
        vector = np.array(gray).flatten()
        X.append(vector)

X = np.array(X)
y = np.array(labels)

print('Oblika X:', X.shape)
print('Oblika y:', y.shape)

# razdelitev podatkov na ucno in testno mnozico
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4321, shuffle=True, stratify=y
)

# ucenje modela odlocitvenega drevesa
clf = DecisionTreeClassifier(random_state=1234)
clf.fit(X_train, y_train)

# izris prvih ravni odlocitvenega drevesa
plt.figure(figsize=(16, 8))
plot_tree(clf, max_depth=2, class_names=clf.classes_, filled=True, fontsize=10)
plt.show()

# napovedi in izracun metrik na testni mnozici
y_pred = clf.predict(X_test)

print('Tocnost (Accuracy):', accuracy_score(y_test, y_pred))
print('Utezena F1-vrednost (Weighted F1-score):', f1_score(y_test, y_pred, average='weighted'))
print('Utezena preciznost (Weighted Precision):', precision_score(y_test, y_pred, average='weighted'))
print('Utezen priklic (Weighted Recall):', recall_score(y_test, y_pred, average='weighted'))

# izracun in graficni prikaz matrike zmede
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=categories, cmap='Blues', ax=ax)
plt.title('Matrika zmede')
plt.show()

# ucenje knn modelov z razlicnimi k in primerjalna tabela z odlocitvenim drevesom
rezultati = [{
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Weighted F1': f1_score(y_test, y_pred, average='weighted'),
    'Weighted Precision': precision_score(y_test, y_pred, average='weighted'),
    'Weighted Recall': recall_score(y_test, y_pred, average='weighted')
}]

for k in [3, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    p = knn.predict(X_test)
    rezultati.append({
        'Model': f'KNN (k={k})',
        'Accuracy': accuracy_score(y_test, p),
        'Weighted F1': f1_score(y_test, p, average='weighted'),
        'Weighted Precision': precision_score(y_test, p, average='weighted'),
        'Weighted Recall': recall_score(y_test, p, average='weighted')
    })

tabela = pd.DataFrame(rezultati)
print(tabela)

# graficna primerjava rezultatov z locenimi stolpicnimi grafi
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.bar(tabela['Model'], tabela['Accuracy'])
plt.title('Accuracy')
plt.xticks(rotation=45)

plt.subplot(1, 4, 2)
plt.bar(tabela['Model'], tabela['Weighted F1'])
plt.title('Weighted F1')
plt.xticks(rotation=45)

plt.subplot(1, 4, 3)
plt.bar(tabela['Model'], tabela['Weighted Precision'])
plt.title('Weighted Precision')
plt.xticks(rotation=45)

plt.subplot(1, 4, 4)
plt.bar(tabela['Model'], tabela['Weighted Recall'])
plt.title('Weighted Recall')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
