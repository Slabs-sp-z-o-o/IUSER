# language: pl

Właściwość: tworzenie modeli z różnymi algorytmami dla wskazanych węzłów

#  Założenia: środowisko testowe nie jest mocno obciążone
#    Zakładając nie więcej niż 10 zadań aktualnie w kolejce

  Scenariusz: tworzenie modelu dla nieistniejącego węzła zakończone błędem
    Mając niezdefiniowany węzeł w lokacji o numerze 1234
    Jeśli zlecę utworzenie modelu autoarima dla zużycia
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model nie będzie gotowy
    i komunikat błędu będzie zawierał "Invalid node ID"

  Szablon scenariusza: tworzenie modelu zakończone sukcesem
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu <algorytm> dla <prognoza>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy

    Przykłady: proste
      | algorytm   | prognoza  |
      | autoarima  | produkcji |
      | autosarima | produkcji |
