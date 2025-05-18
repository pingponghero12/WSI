# Sprawozdanie – Optymalny rozwiązywacz puzli 8 i 15 (A*)

## Opis zadania

Zaimplementować solver 8 i 15 puzzel, który znajduje ścieżkę optymalną pod względem liczby ruchów (minimalna liczba przesunięć kafelków). Rozwiązanie bazuje na algorytmie A* z różnymi heurystykami.

## Heurystyki

- Misplaced Tiles:** Liczba kafelków, które nie są na swoich miejscach w stosunku do stanu końcowego.
- Odległość w metryce Manhattan: Suma odległości w pionie i poziomie każdego kafelka od jego miejsca docelowego.
- Konflikty liniowe (Linear Conflicts): Odległość Manhattan powiększona o dodatkowe 2 ruchy za każdą parę kafelków w konflikcie (czyli są w tej samej linii, ale w złej kolejności względem siebie).
- **Pattern Database (PDB):** Heurystyka oparta na wstępnie obliczonej bazie dystansów dla podzbioru kafelków (np. 6 dla 4x4). Wynik to suma wartości z PDB i odległości Manhattan dla pozostałych kafelków.

## Generowanie permutacji

Permutacje są generowane przez losowe przestawienie kafelków (z wyłączeniem zera, które umieszczane jest na końcu) i sprawdzenie czy wylosowana plansza jest rozwiązywalna wg znanego wzoru (parzystość inwersji i pozycja pustego pola – zgodnie z rozmiarem planszy). Użyto algorytmu Fishera-Yatesa (std::shuffle) i źródła losowości std::mt19937, co zgrubsza zapewnia rozkład jednostajny.

```cpp
std::vector<int> generate_initial_state(int grid_size, const std::vector<int>& goal_state) {
    int num_tiles = grid_size * grid_size;
    std::vector<int> board(num_tiles), tiles;

    for (int i = 1; i < num_tiles; ++i) tiles.push_back(i);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    do {
        std::shuffle(tiles.begin(), tiles.end(), g);
        for (int i = 0; i < num_tiles - 1; ++i) board[i] = tiles[i];
        board[num_tiles - 1] = 0;
    } while (!is_solvable(board, grid_size, goal_state));
    return board;
}
```

## Opis działania algorytmu

1. Stan gry to permutacja tablicy `board` o rozmiarze N.
2. Graf połączeń – każdy stan ma krawędzie do stanów uzyskanych przez przesunięcie pustego pola w 4 kierunkach (jeśli to możliwe).
3. Algorytm A*:
    - Przechowuje w kolejce priorytetowej stany według f = g + h.
    - g – liczba ruchów od startu, h – wartość heurystyki.
    - Po osiągnięciu stanu końcowego rekonstruuje ścieżkę cofając się po rodzicach.
4. W celu efektywnej rekonstrukcji ścieżki, mapujemy każdy stan na jego rodzica przy generowaniu, co umożliwia O(1) dostęp przy cofnięciu się do początku.
5. Sprawdzanie odwiedzonych stanów – set/hashset, żeby nie przetwarzać dwa razy tych samych stanów.

## Złożoność czasowa i pamięciowa

- Czasowa: Zależna od liczby odwiedzonych stanów (przy A* z dopuszczalną heurystyką jest to podzbiór wszystkich stanów). Koszt pojedynczego kroku to generacja sąsiadów (O(1)) oraz obliczenie heurystyki (O(N) dla Manhattan, O(N^2) dla Linear Conflicts).
- Pamięciowa: Przechowujemy wszystkie odwiedzone stany (O(V)), kolejkę priorytetową (O(V)), mapę do rekonstrukcji ścieżki (O(V)). Dla 8-puzzle to ~200 tys. stanów, dla 15-puzzle nawet miliardy (ale przy dobrej heurystyce liczba odwiedzonych jest rzędu milionów).

## Mechanizmy przyspieszające

- Wersja z Pattern Database korzysta z wstępnie obliczonego PDB, który ładowany jest z pliku, a w razie braku pliku generowany i zapisywany na dysk. Aczkowiek jak wychodzi z testów, dla 6 kafelków mamy 2GB plik, którego władowanie itp zawsze sprawia że mamy powyżej 30 sek. Więc jednak się nie opłaca.
- Możliwość podania planszy wejściowej przez argumenty programu (do testów konkretnych przypadków).

## Eksperymenty i wykresy

Skrypt w bashu, który uruchamia solver 1000 razy dla układanki 3x3 z każdą heurystyką i zapisuje wyniki do pliku CSV. Wyniki anlizowane w pythonie. Skrypt nie działa dla 4x4 bo tam dla manhattan przy złych wiatrach są stany których mój komputer z 20GB ramu nie uciąga i zaczyna przeładowywać do swapu, co jest równoważne liczeniu dramatycznie wolno.

## Wnioski

- Wszystkie implementacje zwracają rozwiązania optymalne (najkrótsze możliwe).
- Heurystyka Manhattan istotnie przyspiesza rozwiązanie względem Misplaced Tiles, a Linear Conflicts jeszcze bardziej.
- Pattern Database jest słabo efektywna dla 15-puzzle, bo wymaga dużej ilości RAM i czasu na przygotowanie bazy oraz potem jej wgrania.
- Liczba odwiedzonych stanów rośnie wykładniczo z długością rozwiązania i rozmiarem planszy(w oczywisty sposób).
- Algorytm A* z dobrze dobraną heurystyką pozwala rozwiązać 8-puzzle bardzo szybko, a 15-puzzle dla prostszych przypadków w też szybko, w szczególności z użyciem linear conflicts. 
- Dla 15-puzzle raz na jakiś czas(na oko co 10) zdaża sie przypadek tak duży że przeładowuje 20GB RAMu u mnie na kompie co praktycznie oznacza że nie mogę go policzyć bo tak wolno idzie.
