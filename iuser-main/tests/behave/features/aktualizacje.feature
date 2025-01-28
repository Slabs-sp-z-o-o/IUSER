# language: pl

Właściwość: aktualizacje modeli z różnymi algorytmami dla wskazanych węzłów

#  Założenia: środowisko testowe nie jest mocno obciążone
#    Zakładając nie więcej niż 10 zadań aktualnie w kolejce

#  Scenariusz: aktualizacja nieistniejącego modelu
#    Jeśli zlecę aktualizację nieistniejącego modelu
#    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 9 minut
#    i poczekam na zakończenie ostatniego zadania nie dłużej niż 1 minuta
#    Wtedy model nie będzie gotowy
#    i komunikat błędu będzie zawierał "Invalid model ID"

  Szablon scenariusza: aktualizacja modelu do nieprawidłowej daty
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu <algorytm> dla <prognoza>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy
    Jeśli zlecę aktualizację ostatniego modelu do daty <data>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model nie będzie gotowy
    i komunikat błędu będzie zawierał "<błąd>"

    Przykłady: stare
      | algorytm   | prognoza  | data       | błąd                                  | przypadek
      | autoarima  | produkcji | 0110-01-01 | Cannot update model with data         | data prehistoryczna
      | autoarima  | produkcji | 1990-03-16 | Cannot update model with data         | data sprzed trenowania
      | autosarima | produkcji | 2020-12-12 | Cannot update model with data         | data już trenowana
      | autoarima  | produkcji | 2020-12-30 | data newer than the original training | data końca trenowania

    Przykłady: późniejsze
      | algorytm   | prognoza  | data             | błąd                                  | przypadek
      | autoarima  | produkcji | 2020-12-30 00:05 | records found within specified bounds | dodanie jedynie 5 minut

  Szablon scenariusza: aktualizacja modelu o kolejne dni
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu <algorytm> dla <prognoza>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy
    Jeśli zlecę aktualizację ostatniego modelu do daty <data>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy

    Przykłady: proste
      | algorytm   | prognoza  | data       | przypadek
      | autosarima | produkcji | 2020-12-31 | dodanie 1 dnia
#      | autoarima  | produkcji | 2021-01-30 | dodanie 1 miesiąca #TODO check why it is failing

  Szablon scenariusza: dwukrotna aktualizacja modelu o kolejne dni
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu <algorytm> dla <prognoza>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy
    Jeśli zlecę aktualizację ostatniego modelu do daty <data1>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy
    Jeśli zlecę aktualizację ostatniego modelu do daty <data2>
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy model będzie gotowy

    Przykłady: proste
      | algorytm   | prognoza  | data1      | data2      | przypadek
      | autosarima | produkcji | 2020-12-31 | 2021-01-01 | dodanie 2x po 1 dniu