# Una prima prova locale

Hermes Agent + Ollama, principalmente su Linux, anche su macOS. Materiali del talk "Liberi di usare l'AI", Linux Day Enna, 24 ottobre 2026.

Guida verificata sulle fonti ufficiali il **23 ottobre 2026**. I comandi di configurazione sono stati controllati anche con la CLI Hermes disponibile durante la preparazione. Non è una certificazione di compatibilità con qualsiasi computer: modello, memoria, driver e versioni contano.

## 1. Prima di installare

Servono un terminale, spazio libero e una connessione per scaricare software, dipendenze e pesi. Non devi rifare il training: Ollama esegue un modello già preparato, Hermes gestisce conversazione e strumenti.[1][10][11]

- **Linux:** i comandi di sistema qui sotto sono per Ubuntu/Debian con systemd. Per altre distribuzioni segui la pagina ufficiale Linux di Ollama.[3]
- **macOS:** Ollama richiede macOS Sonoma 14 o successivo. I Mac Apple Silicon possono usare la GPU; i Mac Intel usano la CPU.[4]
- **Hardware:** controlla RAM, GPU e relativi driver. La dimensione del download NON è la memoria necessaria: contesto e altri programmi occupano ulteriore spazio in RAM/VRAM.[5][8]

Su Ubuntu/Debian prepara le utility di base:

```bash
sudo apt update
sudo apt install -y curl git unzip ca-certificates build-essential
```

Su macOS usa Terminale e verifica `git --version`. Se macOS propone gli strumenti da riga di comando Apple, completa l'installazione prima di proseguire.

**Prima di eseguire uno script scaricato, leggilo e verifica il dominio.** Le pipeline di installazione sotto eseguono codice sul computer. Non anteporre `sudo` all'installazione di Hermes nel suo insieme.

## 2. Installare Ollama

Apri [la pagina Download](https://ollama.com/download) e scegli il tuo sistema.[1]

### Linux

La procedura ufficiale propone:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Su un sistema con systemd controlla il servizio:

```bash
sudo systemctl start ollama
systemctl is-active ollama
ollama -v
curl -fsS http://localhost:11434/api/tags
```

L'ultima richiesta deve restituire JSON, anche se non hai ancora modelli. Il servizio deve restare attivo mentre usi Hermes.[3]

### macOS

Dalla pagina Download scarica l'app per Mac, apri il DMG, trascina Ollama in **Applicazioni** e avviala. Completa l'eventuale richiesta per rendere disponibile il comando `ollama`, poi riapri il terminale.[4]

```bash
ollama -v
curl -fsS http://localhost:11434/api/tags
```

Se l'app o il servizio sono già attivi, **non avviare un secondo server**. Solo se stai usando un'installazione manuale senza servizio/app, puoi tenere `ollama serve` aperto in un terminale separato.[3][4]

## 3. Scegliere e scaricare un modello

Apri [il catalogo Ollama](https://ollama.com/search). Cerca un modello **locale**, leggi la scheda e i tag, controlla licenza, hardware e supporto **Tools**. Un modello cloud non è un insieme di pesi eseguito sul tuo PC.[2][14]

Per questa guida usiamo **`qwen3.5:9b`**, un tag pubblico con supporto agli strumenti e contesto dichiarato di 256K. Non è una promessa di qualità o velocità, ne' il requisito per riprodurre esattamente la macchina del relatore.[6]

| Esempio locale | Download indicativo al controllo | Per orientarsi |
| --- | --- | --- |
| `qwen3.5:9b` | 6.6-7.6 GB secondo il backend | Esempio usato nei comandi di questa guida. |
| `qwen3:4b` | 2.5 GB | Alternativa più piccola; controlla comunque qualità e memoria con il contesto richiesto. |

Entrambe le schede consultate indicano Tools e contesto 256K. I tag e i backend possono cambiare: ricontrolla le pagine prima del download. **Questi GB non sono un requisito di RAM.**[6][7][8]

```bash
ollama pull qwen3.5:9b
ollama list
ollama run qwen3.5:9b
```

Nella chat di Ollama prova: "Traduci in inglese: Benvenuti al Linux Day". Esci con `/bye`.

Hai verificato solo l'inferenza testuale, non ancora la lettura dei file o il funzionamento di un agente. Il badge Tools non significa che `ollama run` esegua da solo comandi o legga documenti per te.

## 4. Preparare il contesto per Hermes

**Non saltare questo passaggio.** Hermes attuale richiede almeno **64.000 token** per il contesto agentico. Impostare una finestra piccola solo per risparmiare memoria può impedire l'avvio. Anche Ollama deve allocare il contesto corrispondente; il massimo dichiarato nella scheda non è il contesto realmente attivo.[8][11]

Crea una cartella di lavoro nuova. Se `~/linuxday` esiste già, usa un altro nome per non sovrascrivere prove precedenti:

```bash
mkdir -p "$HOME/linuxday"
cd "$HOME/linuxday"
printf 'FROM qwen3.5:9b\nPARAMETER num_ctx 64000\n' > Modelfile.linuxday
ollama create linuxday-qwen64k -f Modelfile.linuxday
ollama run linuxday-qwen64k
```

`linuxday-qwen64k` è **un nome locale creato da questi comandi**, non un modello da cercare nel catalogo. Il Modelfile configura l'esecuzione, non fa training.[9]

Prova una richiesta breve. Mentre il modello è caricato, in un altro terminale esegui:

```bash
ollama ps
```

Controlla la colonna **CONTEXT**: deve indicare almeno 64000. Poi chiudi la chat Ollama con `/bye`. Un contesto più ampio usa più memoria.[8]

**Se non entra in memoria:** puoi provare `qwen3:4b`, scaricandolo e sostituendo `qwen3.5:9b` nella riga `FROM`, poi ricreare il modello locale. Non abbassare il requisito di Hermes fingendo che sia equivalente: se anche la configurazione più piccola non funziona, resta alla prova testuale Ollama o usa una macchina più adatta.[7][11]

## 5. Installare Hermes Agent

Linux e macOS usano lo stesso installer ufficiale:[10]

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Segui le istruzioni, poi **chiudi e riapri il terminale**. In alternativa ricarica il file della tua shell: `source ~/.bashrc` per Bash, oppure `source ~/.zshrc` per Zsh. Verifica:

```bash
hermes --version
```

Se il comando non viene trovato, per la shell corrente prova il percorso del launcher ufficiale:

```bash
export PATH="$HOME/.local/bin:$PATH"
hermes --version
```

Per questo percorso locale non occorre acquistare un servizio LLM o inserire una chiave cloud.[11]

## 6. Collegare Hermes a Ollama

Usiamo un profilo dedicato, **`linuxday`**, senza cambiare il profilo predefinito. Se hai già un profilo con quel nome, scegline uno nuovo e sostituiscilo in TUTTI i comandi. Non usare `--clone`.[12]

```bash
hermes profile create linuxday --no-skills --no-alias
hermes -p linuxday config unset model
hermes -p linuxday config set providers '{}'
hermes -p linuxday config unset custom_providers
hermes -p linuxday config set model.provider custom
hermes -p linuxday config set model.base_url http://localhost:11434/v1
hermes -p linuxday config set model.default linuxday-qwen64k
hermes -p linuxday config set model.api_key ollama
hermes -p linuxday config set model.api_mode chat_completions
hermes -p linuxday config set model.context_length 64000
hermes -p linuxday config set model.ollama_num_ctx 64000
hermes -p linuxday config set terminal.backend local
hermes -p linuxday config set approvals.mode manual
```

`custom` indica un endpoint compatibile; l'indirizzo punta al server **sullo stesso computer**, non a un servizio remoto. Il valore `ollama` della chiave è una stringa fittizia per l'endpoint locale, non una credenziale. I valori di contesto devono essere coerenti con il Modelfile.[11]

Per evitare un ripiego involontario su altri modelli e mantenere locale anche la compressione del contesto, completa la configurazione:

```bash
hermes -p linuxday config set fallback_providers '[]'
hermes -p linuxday config unset fallback_model
hermes -p linuxday config set auxiliary.title_generation.enabled false
hermes -p linuxday config set auxiliary.compression.provider custom
hermes -p linuxday config set auxiliary.compression.base_url http://localhost:11434/v1
hermes -p linuxday config set auxiliary.compression.model linuxday-qwen64k
hermes -p linuxday config set auxiliary.compression.api_key ollama
hermes -p linuxday config set curator.enabled false
hermes -p linuxday config set security.allow_lazy_installs false
```

Un profilo separa configurazione e sessioni, **non è una sandbox** e non garantisce da solo l'assenza di credenziali condivise o di accesso ai tuoi file. I comandi qui configurano il percorso locale, ma non sono un firewall. La modalità `manual` riguarda le approvazioni dei comandi: non richiede una conferma per ogni lettura o scrittura di file.[12][13]

Controlla i valori senza mostrare chiavi o token personali:

```bash
hermes -p linuxday config get model.provider
hermes -p linuxday config get model.base_url
hermes -p linuxday config get model.default
hermes -p linuxday config get model.context_length
hermes -p linuxday fallback list
```

## 7. Scaricare il materiale e preparare il lettore

Lo ZIP contiene documenti fittizi, prompt e questa guida. Non contiene un risultato già compilato. Dalla cartella di lavoro:

```bash
cd "$HOME/linuxday"
curl -fL https://mlaquatra.me/talks/2026-linux-day/materiali-linuxday-2026.zip -o materiali.zip
unzip materiali.zip
cd linuxday-2026
```

Non scegliere "sovrascrivi tutto" se stai ripetendo una prova e hai già modificato i file. La cartella `output/` del pacchetto parte vuota.

Per evitare che la prima lettura PDF richieda una dipendenza imprevista, prepariamo il lettore locale incluso. Prima controlla:

```bash
uv --version
```

Se `uv` non è nel PATH, segui [l'installazione ufficiale di uv](https://docs.astral.sh/uv/getting-started/installation/). Su Linux/macOS l'installer ufficiale è:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Riapri il terminale se necessario, poi torna in `~/linuxday/linuxday-2026`. Anche questo comando esegue uno script scaricato: controllalo prima.[16]

Prepara e prova l'estrazione, senza installare pacchetti nell'ambiente di Hermes:

```bash
uv run --no-project --with pymupdf python leggi-documenti.py
```

Devi vedere tre blocchi: `menu.pdf`, `ricette.docx`, `dispensa.csv`. Questo piccolo script legge il PDF con PyMuPDF e il DOCX/CSV con Python; non calcola la spesa e non invia il contenuto a un servizio. L'esecuzione iniziale può scaricare Python o dipendenze nell'ambiente/cache di uv.[15]

Dopo la preparazione, verifica anche:

```bash
uv run --offline --no-project --with pymupdf python leggi-documenti.py
```

Se fallisce, prepara cio' che manca prima di disconnettere la rete. Questo lettore gestisce i documenti inclusi: non è un sistema OCR per qualsiasi PDF.[15]

## 8. Prima prova: tradurre il menu

Dalla cartella `linuxday-2026`, apri una nuova chat:

```bash
hermes -p linuxday chat -t file
```

Apri `esempio-traduzione.txt` con un editor e incolla il suo contenuto. Per questa richiesta non occorre usare strumenti: il menu è già nel prompt. L'opzione `-t file` lascia comunque disponibili gli strumenti per i file; non è una modalità tecnicamente "senza strumenti".

Controlla traduzione, piatti e informazioni sugli allergeni. Non usare una risposta automatica come unica verifica per esigenze alimentari reali.

## 9. Seconda prova: leggere i file e scrivere la lista

Chiudi la prima chat con `/exit`. Nella stessa cartella avvia una chat nuova con i soli gruppi di strumenti necessari:

```bash
hermes -p linuxday chat -t terminal,file
```

Incolla `esempio-analisi-file.txt`. Il prompt usa percorsi relativi: gli input sono in `cena/` e il risultato va in `output/lista-spesa.md`.

Per un percorso di lettura riproducibile puoi aggiungere al messaggio:

> Per estrarre il testo dei tre documenti, puoi eseguire il lettore locale incluso con `uv run --offline --no-project --with pymupdf python leggi-documenti.py`. Usa i dati effettivamente letti e non modificare gli input.

Leggi le eventuali richieste di esecuzione prima di approvarle. **Non usare `--yolo`.** Il terminale può eseguire codice e l'agente ha i permessi dell'utente del sistema operativo: un elenco ristretto di strumenti o una cartella dedicata non limita automaticamente il filesystem o la rete.[13]

Alla fine, verifica davvero il file:

```bash
cat output/lista-spesa.md
```

Confrontalo con i documenti, non solo con l'ultima frase della chat:

- Ha letto tutti i tre input?
- Ha mantenuto le dosi per quattro persone?
- Ha sottratto le scorte, senza comprare tutto da zero?
- Ha escluso i prodotti presenti in dispensa ma estranei alle ricette?
- Ha segnalato le quantità non determinabili, invece di inventarle?
- Il file esiste ed è comprensibile?

Modifica una copia della dispensa e ripeti in una nuova chat. Non è garantito che un modello capace di chiamare strumenti riesca correttamente al primo tentativo: questa verifica è parte dell'esercizio.

## 10. Locale, offline e problemi comuni

**Locale:** l'endpoint usato per il modello è `localhost`. **Offline:** l'intero percorso deve riuscire senza connessioni esterne. Non sono la stessa cosa. Prima scarica software, dipendenze e pesi; poi ripeti l'esercizio disattivando la rete esterna e lasciando disponibile localhost.[11][14][15]

Non aggiungere ricerca web, browser, modelli cloud, gateway o servizi remoti a questa prova. Per documenti personali usa un utente dedicato, una VM o un ambiente con accessi limitati. Anche cronologia, log e file salvati localmente meritano attenzione.[13]

| Problema | Cosa controllare |
| --- | --- |
| `Connection refused` | Ollama non è avviato. Apri l'app Mac o controlla il servizio Linux; riprova `/api/tags`. |
| Modello non trovato | `ollama list`; verifica di aver creato `linuxday-qwen64k` e che Hermes usi esattamente quel nome. |
| Errore sul contesto | Modelfile a 64000, impostazioni Hermes coerenti, valore reale in `ollama ps`. Non confondere massimo del catalogo e contesto attivo. |
| Memoria insufficiente o risposte molto lente | Controlla GPU/driver, altri programmi e contesto. Prova un modello più piccolo compatibile; non promettere che la CPU abbia la stessa velocità della GPU. |
| Il PDF non viene letto | Prova `leggi-documenti.py` dalla cartella principale. Il PDF incluso ha testo; altri PDF possono richiedere OCR. |
| La chat risponde ma il file non esiste | Non considerare concluso il compito. Controlla strumenti abilitati, permessi, percorso e chiamate effettivamente eseguite. |
| Una chiave di configurazione viene rifiutata | Controlla `hermes --version`, `hermes config set --help` e la documentazione attuale. Non usare `--force` per nascondere una differenza di versione. |

Per approfondire torna alla [pagina risorse](https://mlaquatra.me/talks/2026-linux-day/risorse.html). Annota sistema, RAM/GPU, versioni, modello, contesto effettivo e limiti osservati: una prova riuscita su una macchina non è una garanzia per tutte le altre.

## Sources

[1] https://ollama.com/download - Download Ollama
[2] https://ollama.com/search - Catalogo Ollama
[3] https://docs.ollama.com/linux - Ollama su Linux
[4] https://docs.ollama.com/macos - Ollama su macOS
[5] https://docs.ollama.com/gpu - GPU supportate
[6] https://ollama.com/library/qwen3.5:9b - Qwen3.5 9B in Ollama
[7] https://ollama.com/library/qwen3:4b - Qwen3 4B in Ollama
[8] https://docs.ollama.com/context-length - Contesto Ollama
[9] https://docs.ollama.com/modelfile - Modelfile
[10] https://hermes-agent.nousresearch.com/docs/getting-started/installation - Installare Hermes
[11] https://hermes-agent.nousresearch.com/docs/guides/local-ollama-setup - Hermes con Ollama
[12] https://hermes-agent.nousresearch.com/docs/user-guide/profiles - Profili Hermes
[13] https://hermes-agent.nousresearch.com/docs/user-guide/security - Sicurezza Hermes
[14] https://docs.ollama.com/faq - FAQ Ollama
[15] https://raw.githubusercontent.com/astral-sh/uv/main/docs/guides/scripts.md - Eseguire script con uv
[16] https://raw.githubusercontent.com/astral-sh/uv/main/docs/getting-started/installation.md - Installare uv
