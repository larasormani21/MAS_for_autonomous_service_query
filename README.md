# Sistema Multi-Agente per l’Interrogazione Autonoma di Servizi
Il progetto “Sistema Multi-Agente per l’Interrogazione Autonoma di Servizi:
esplorazione delle tecnologie disponibili” nasce con l’obiettivo di analizzare e
sperimentare le funzionalità offerte dalle più recenti tecnologie di Large Language Models (LLM) e dei framework per sistemi multi-agente.

L’idea centrale consiste nello sviluppo di un prototipo capace di interrogare
autonomamente servizi web, evitando la necessità di un intervento manuale nella
definizione delle chiamate API.

Il contesto applicativo è costituito da un repository eterogeneo di descrizioni
di servizi, alcune conformi a formati standard e altre non strutturate. In tale
scenario, il sistema multi-agente è progettato per:
* Interpretare la richiesta dell’utente,
* Identificare e selezionare i servizi disponibili pi`u pertinenti,
* Formulare ed eseguire le chiamate API,
* Restituire una risposta sintetica e leggibile.

L’obiettivo principale del lavoro non è la realizzazione di una soluzione pronta
all’utilizzo in contesti produttivi, bensì l’avvio di un percorso di esplorazione
critica delle tecnologie disponibili, al fine di comprenderne i punti di forza ed i
limiti, motivare le scelte progettuali ed individuare i margini di miglioramento.
Tale percorso permette di delineare possibili evoluzioni future verso scenari più
complessi e applicazioni utilizzabili in modo concreto.
 
## Struttura del progetto

```
.
├── agents/                
│   ├── converter.py
│   ├── executor.py
│   ├── feedback.py
│   ├── indexer.py
│   ├── retriever.py
│   └── extractor.py
├── services_descriptions/ # Esempi di file JSON/YAML/HTML dei servizi
├── config.py              
├── .env                   # Parametri configurabili dall'utente
├── graph.py               
├── state.py               
└── README.md
```