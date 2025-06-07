# Sprawozdanie Lista 4

Wszystkie wykresy w 'img'.

### Zadanie 1
Przeprowadzona analiza potwierdza, że algorytm k-średnich (k-means) dobrze sprawdza się w przypadku klastrowania zbioru danych MNIST. Już przy zastosowaniu 10 klastrów możemy zaobserwować wyraźne reprezentacje wszystkich cyfr, a także interesujące struktury pośrednie, takie jak hybrydy cyfr 4-9 czy 3-8.

Macierz przypisań wskazuje na pewne nakładanie się klastrów, co jest naturalnym zjawiskiem wynikającym z podobieństwa niektórych cyfr (np. 3 i 8, 5 i 6, 7 i 1). Mimo to, ogólna jakość klastrowania pozostaje na wysokim poziomie, co potwierdza skuteczność algorytmu dla tego typu danych.

Zwiększenie liczby klastrów do 15, 20 i 30 prowadzi do lekkiej poprawy separacji, redukując zjawisko nakładania się. Co więcej, przy większej liczbie klastrów możemy zaobserwować subtelne różnice w stylu pisania tej samej cyfry - na przykład różne warianty cyfry 7 (z poprzeczką i bez) zostają przypisane do odrębnych klastrów, co stanowi cenne spostrzeżenie z perspektywy analizy wzorców pisma odręcznego.

Centroidy klastrów wyraźnie przypominają cyfry, co dodatkowo potwierdza trafność klastrowania. Metoda k-średnich radzi sobie dobrze z danymi MNIST dzięki globalnemu podejściu do kształtu przestrzeni danych oraz dzięki metryce odległości euklidesowej, która dobrze uchwytuje podobieństwo między obrazami cyfr.

```

============================================================
ANALYSIS FOR 10 CLUSTERS
============================================================

Performing k-means clustering for 10
Best inertia after 10 trials: 8641481.28

Clustering Metrics:
Inertia: 8641481.28
Adjusted Rand Index: 0.300
Normalized Mutual Information: 0.413

============================================================
ANALYSIS FOR 15 CLUSTERS
============================================================

Performing k-means clustering for 15
Best inertia after 10 trials: 8261883.52

Clustering Metrics:
Inertia: 8261883.52
Adjusted Rand Index: 0.291
Normalized Mutual Information: 0.451

============================================================
ANALYSIS FOR 20 CLUSTERS
============================================================

Performing k-means clustering for 20
^[Best inertia after 10 trials: 8015543.94

Clustering Metrics:
Inertia: 8015543.94
Adjusted Rand Index: 0.302
Normalized Mutual Information: 0.467

============================================================
ANALYSIS FOR 30 CLUSTERS
============================================================

Performing k-means clustering for 30
Best inertia after 10 trials: 7553578.38

Clustering Metrics:
Inertia: 7553578.38
Adjusted Rand Index: 0.295
Normalized Mutual Information: 0.486
```

### Zadanie 2
W przeciwieństwie do k-średnich, algorytm DBSCAN okazuje się być nieodpowiedni dla zbioru MNIST, głównie z powodu problemu znanego jako "klątwa wymiarowości". Przestrzeń 784-wymiarowa (28x28 pikseli) powoduje, że pojęcie gęstości, kluczowe dla DBSCAN, traci swoją skuteczność.

Próby redukcji wymiarowości, czy to poprzez techniki takie jak MaxPooling, czy zastosowane w naszym przypadku PCA, nie przynoszą zadowalających rezultatów. Niezależnie od doboru parametrów eps (promień sąsiedztwa) i min_samples (minimalna liczba sąsiadów), DBSCAN ma tendencję do tworzenia jednego dominującego klastra, klasyfikując pozostałe punkty jako szum.

Szczególnie interesujące jest to, że algorytm wykazuje silną preferencję dla cyfry 1, prawdopodobnie ze względu na jej prostą strukturę i podobieństwo do fragmentów innych cyfr. W przestrzeni wysokowymiarowej cyfra 1 może być "bliżej" innych cyfr niż mogłoby się intuicyjnie wydawać, co prowadzi do tworzenia niejednorodnych klastrów.

Parametryzacja DBSCAN jest również problematyczna dla tego zbioru danych - zbyt małe wartości eps prowadzą do klasyfikacji większości punktów jako szumu, natomiast zbyt duże wartości skutkują tworzeniem jednego niejednorodnego klastra. Znalezienie optymalnego balansu okazuje się niezwykle trudne nawet przy zastosowaniu parametrica, jest to w szczególności trudne z powodu wielu metryk do oceny klastrowania.

Podsumowując, dla zadań klastrowania danych obrazowych o wysokiej wymiarowości, takich jak MNIST, metoda k-średnich znacząco przewyższa DBSCAN pod względem jakości wyników i łatwości interpretacji utworzonych klastrów.

```
Running DBSCAN with eps=8.0, min_samples=5
Clusters: 10
Noise:2615
Purity: 0.9075
Accuracy: 0.1519
Error rate: 0.8481
Silhouette: 0.0394
Combined score: 0.5689

Detailed cluster breakdown:
----------------------------------------------------------------------
Cluster   Size    Dominant  Purity    Distribution
----------------------------------------------------------------------
Noise     2615    N/A       N/A       2:612, 3:400, 0:363
0         9326    1         0.14671:1368, 9:1099, 7:1043
1         9       2         1.00002:9
2         14      0         0.92860:13, 6:1
3         8       0         1.00000:8
4         3       2         1.00002:3
5         4       2         1.00002:4
6         5       3         1.00003:5
7         4       8         1.00008:4
8         9       2         1.00002:9
9         3       8         1.00008:3

Digit capture analysis:
----------------------------------------------------------------------
Digit   Total   Clustered Noise   Capture   Accuracy
----------------------------------------------------------------------
0       1175    812       363     0.6911    0.0259
  Dominant in clusters: 2(0.93), 3(1.00)
1       1377    1368      9       0.9935    1.0000
  Dominant in clusters: 0(0.15)
2       1173    561       612     0.4783    0.0446
  Dominant in clusters: 1(1.00), 4(1.00), 5(1.00), 8(1.00)
3       1227    827       400     0.6740    0.0060
  Dominant in clusters: 6(1.00)
4       1125    930       195     0.8267    0.0000
5       1095    838       257     0.7653    0.0000
6       1176    972       204     0.8265    0.0000
7       1275    1043      232     0.8180    0.0000
8       1159    935       224     0.8067    0.0075
  Dominant in clusters: 7(1.00), 9(1.00)
9       1218    1099      119     0.9023    0.0000
Total samples: 12000
```
