# language: pl

Właściwość: zarządzanie kolejką zleconych treningów modeli i prognoz

#  Założenia: środowisko testowe nie jest mocno obciążone
#    Zakładając nie więcej niż 10 zadań aktualnie w kolejce

  Scenariusz: natychmiastowe anulowanie zadania
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu autoarima dla produkcji
    Jeśli anuluję ostatnie zadanie
    i poczekam 4 sekundy
    Wtedy status ostatniego zadania będzie równy CANCELLED/STARTED

  Scenariusz: anulowanie trwającego zadania
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu autoarima dla produkcji
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    Wtedy status ostatniego zadania będzie równy STARTED
    Jeśli anuluję ostatnie zadanie
    i poczekam 4 sekundy
    Wtedy status ostatniego zadania będzie równy CANCELLED/STARTED

  Szablon scenariusza: anulowanie zakończonego zadania
    Mając <stan> węzeł w lokacji o numerze <lokacja>
    Jeśli zlecę utworzenie modelu autoarima dla produkcji
    oraz poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    Wtedy status ostatniego zadania będzie równy <status>
    Jeśli anuluję ostatnie zadanie
    i poczekam 4 sekundy
    Wtedy status ostatniego zadania będzie równy <status>

    Przykłady: błąd
      | stan            | lokacja | status  |
      | niezdefiniowany | 1234  | FAILURE |

    Przykłady: gotowe
      | stan         | lokacja | status  |
      | zdefiniowany | 9999     | SUCCESS |

  Scenariusz: anulowanie anulowanego zadania
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu autoarima dla produkcji
    ale anuluję ostatnie zadanie
    Wtedy status ostatniego zadania będzie równy CANCELLED/STARTED
    Jeśli anuluję ostatnie zadanie
    i poczekam 4 sekundy
    Wtedy status ostatniego zadania będzie równy CANCELLED/STARTED

  Scenariusz: kolejkowanie 3 zadań
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu autoarima dla produkcji
    oraz zlecę utworzenie modelu autoarima dla produkcji
    oraz zlecę utworzenie modelu autoarima dla produkcji
    Wtedy statusy wszystkich zadań będą równe PENDING/STARTED
    Jeśli poczekam na uruchomienie ostatniego zadania nie dłużej niż 8 minut
    i poczekam na zakończenie ostatniego zadania nie dłużej niż 4 minuty
    i poczekam 4 sekundy
    Wtedy statusy wszystkich zadań będą równe SUCCESS

  Scenariusz: anulowanie oczekujących 5 zadań
    Mając zdefiniowany węzeł w lokacji o numerze 9999
    Jeśli zlecę utworzenie modelu autoarima dla produkcji
    oraz zlecę utworzenie modelu autoarima dla produkcji
    oraz zlecę utworzenie modelu autoarima dla produkcji
    oraz zlecę utworzenie modelu autoarima dla produkcji
    oraz zlecę utworzenie modelu autoarima dla produkcji
    Wtedy statusy wszystkich zadań będą równe PENDING/STARTED
    Jeśli anuluję wszystkie zadania
    i poczekam 4 sekundy
    Wtedy statusy wszystkich zadań będą równe STARTED/CANCELLED/PENDING
