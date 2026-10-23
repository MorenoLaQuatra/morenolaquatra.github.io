# Liberi di usare l'AI

Materiali del talk di Moreno La Quatra, Linux Day Enna, 24 ottobre 2026.

- Slide: https://mlaquatra.me/talks/2026-linux-day/
- Risorse: https://mlaquatra.me/talks/2026-linux-day/risorse.html
- Guida passo passo: https://mlaquatra.me/talks/2026-linux-day/guida.html

## Cosa contiene il pacchetto

```text
linuxday-2026/
  README.md
  GUIDA.md
  esempio-traduzione.txt
  esempio-analisi-file.txt
  leggi-documenti.py
  cena/
    menu.pdf
    ricette.docx
    dispensa.csv
  output/                  # inizialmente vuota
```

I tre documenti della cena sono gli input usati nella demo. Il prompt sui file
usa percorsi relativi, invece dei percorsi del computer del relatore.
Non ci sono pesi dei modelli, programmi da installare o una soluzione gia' pronta.

## Primo esercizio: traduzione

Apri `esempio-traduzione.txt` e incolla il suo contenuto in una nuova chat.
Verifica il senso del menu e che le informazioni sugli allergeni siano rimaste.
Non servono strumenti per questo primo esercizio.

## Secondo esercizio: documenti

Segui prima `GUIDA.md`. Avvia Hermes dalla cartella `linuxday-2026`, non dalla
cartella `cena`, e incolla `esempio-analisi-file.txt`.
Il programma deve leggere tutti i tre file in `cena/` e scrivere solo in
`output/lista-spesa.md`. Controlla i documenti letti, i calcoli e il file salvato.

`leggi-documenti.py` e' un lettore di supporto: estrae in locale il testo di
questo PDF, di questo DOCX e del CSV, senza cambiare gli input. Richiede PyMuPDF
per il PDF. Per preparare e provare il lettore con uv, come spiegato nella guida:

```bash
uv run --no-project --with pymupdf python leggi-documenti.py
```

Dopo aver preparato la cache, la stessa estrazione puo' essere eseguita senza
download:

```bash
uv run --offline --no-project --with pymupdf python leggi-documenti.py
```

L'estrazione non e' una risposta dell'AI e non calcola la lista della spesa.
Se usi un altro lettore, verifica sempre che tutti i documenti siano stati letti.

## Attenzione ai permessi

Sono documenti fittizi, pensati per fare prove senza usare file personali.
Una cartella dedicata e un profilo Hermes separato NON sono una sandbox:
l'agente puo' avere i permessi dell'utente del sistema operativo. Non usare
`--yolo` e controlla le richieste di esecuzione dei comandi.
Per un isolamento tecnico usa un utente dedicato, una VM o un altro ambiente
con accessi ai file e alla rete limitati.

Per usare tutto senza rete devi avere gia' scaricato software, dipendenze e
pesi e verificato la configurazione completa, non solo il modello.
