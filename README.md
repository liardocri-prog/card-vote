# TCG Card Grader Prototype 🃏

Un prototipo in Python basato su **OpenCV** per l'analisi preliminare e la stima dello stato di conservazione (grading amatoriale stile PSA) di carte collezionabili come Pokémon e One Piece.

## Funzionalità
- **Rilevamento e raddrizzamento automatico:** Isola la carta dallo sfondo e corregge la prospettiva.
- **Analisi dei bordi (Whitening):** Rileva l'usura e i punti bianchi lungo i bordi della carta.
- **Stima del punteggio:** Restituisce un voto preliminare basato sui difetti riscontrati.

## Requisiti
- Python 3.8+

## Installazione

1. Clona il repository:
   ```bash
   git clone [https://github.com/tuo-username/nome-repository.git](https://github.com/tuo-username/nome-repository.git)
   cd nome-repository
