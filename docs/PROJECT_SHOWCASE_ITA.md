# Dynamic Target Suite — Guida di Presentazione e Onboarding

> **Destinatari:** Ingegneri, ricercatori o collaboratori che incontrano questo progetto per la prima volta.  
> **Scopo:** Capire cosa fa la suite, cosa avviare per vederla in azione, quale codice è più significativo e quali errori evitare.  
> Per la matematica approfondita e i dettagli architetturali, consulta [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## Indice

1. [Cos'è Dynamic Target Suite?](#1-cosè-dynamic-target-suite)
2. [Avvio Rapido](#2-avvio-rapido)
3. [Mappa del Repository](#3-mappa-del-repository)
4. [Simulazioni da Mostrare](#4-simulazioni-da-mostrare)
   - 4.1 [Diagramma YMD & Handling](#41-diagramma-ymd--handling)
   - 4.2 [Direct Collocation — Traiettoria Ottimale](#42-direct-collocation--traiettoria-ottimale)
   - 4.3 [Sweep di Sensitività 1D Transitoria](#43-sweep-di-sensitività-1d-transitoria)
   - 4.4 [Simulatore Interattivo 14-DOF](#44-simulatore-interattivo-14-dof)
   - 4.5 [Scoring Skidpad & Accelerazione](#45-scoring-skidpad--accelerazione)
   - 4.6 [Risposta in Frequenza 7-Post Rig](#46-risposta-in-frequenza-7-post-rig)
   - 4.7 [Mappe DNA del Veicolo (n1–n6)](#47-mappe-dna-del-veicolo-n1n6)
5. [Sezioni di Codice Più Significative](#5-sezioni-di-codice-più-significative)
   - 5.1 [Pipeline Compilazione ODE 14-DOF con CasADi](#51-pipeline-compilazione-ode-14-dof-con-casadi)
   - 5.2 [Modello Pneumatico Pacejka MF6.1 — Compatibile CasADi](#52-modello-pneumatico-pacejka-mf61--compatibile-casadi)
   - 5.3 [VehicleParams — Unica Fonte di Verità](#53-vehicleparams--unica-fonte-di-verità)
   - 5.4 [Ottimizzatore MDO Globale — Multi-Start SLSQP](#54-ottimizzatore-mdo-globale--multi-start-slsqp)
   - 5.5 [Worker QThread — Pattern di Calcolo Asincrono](#55-worker-qthread--pattern-di-calcolo-asincrono)
6. [Convenzioni Critiche e Insidie](#6-convenzioni-critiche-e-insidie)
7. [Suite di Test — Come Verificare che Tutto Funzioni](#7-suite-di-test--come-verificare-che-tutto-funzioni)
8. [Eseguibile Standalone](#8-eseguibile-standalone)

---

## 1. Cos'è Dynamic Target Suite?

**Dynamic Target Suite** è un ambiente di lavoro ingegneristico integrato per **vetture da corsa Formula SAE / Formula Student elettriche**. Copre l'intero ciclo di messa a punto del veicolo:

```
Parametri Geometrici e Fisici
        │
        ▼
   Ottimizzazione MDO / Trim          ← SLSQP multi-start, OpenMDAO
        │
        ▼
  Mappe di Handling e DNA             ← YMD, Moment Method, archi MAPs
        │
        ▼
  Validazione Transitoria             ← ODE 14-DOF, CasADi + RK4
        │
        ▼
  Simulatore Interattivo Real-Time    ← Thread fisica a 500 Hz, gamepad/FFB
```

**Capacità principali:**

| Dominio | Cosa fa |
|---------|---------|
| **Modelli Veicolo** | 3-DOF (quasi-statico), 7-DOF (imbardata/rollio transitorio), 14-DOF (sospensioni complete + 4 ruote) |
| **Pneumatico** | Pacejka MF6.1 (non-lineare, simbolico, compatibile CasADi), degradazione termica |
| **MDO** | Ottimizzatore SLSQP multi-start, traiettoria ottima con direct collocation (CasADi + IPOPT) |
| **Prove** | Step Steer, Double Lane Change, Skidpad, Accelerazione, Endurance, 7-Post Rig |
| **Analisi** | Diagrammi YMD / Handling, mappe DNA del veicolo, sweep sensitività 1D (21 KPI), propagazione incertezze |
| **Simulatore** | 14-DOF real-time a 250–500 Hz con gamepad / volante force-feedback |

---

## 2. Avvio Rapido

> [!IMPORTANT]
> Usare **sempre** l'interprete dell'ambiente virtuale locale. Non usare mai Python installato globalmente.

```powershell
# 1. Attivare l'ambiente (opzionale, per la shell interattiva)
.\venv\Scripts\Activate.ps1

# 2. Avviare l'applicazione GUI
.\venv\Scripts\python.exe main.py
```

La GUI che si apre è **DYNAMIC TARGET SUITE PRO** — un'interfaccia con 5 workspace:

| Workspace | Scopo |
|-----------|-------|
| **Setup** | Modifica parametri del veicolo (massa, geometria, sospensioni, powertrain) |
| **MDO** | Esegui ottimizzazioni e visualizza frontiere di Pareto |
| **Analisi** | Diagrammi Handling, YMD, mappe DNA, sweep di sensitività |
| **Simulatore** | Avvia il simulatore interattivo 14-DOF |
| **Risultati** | Dashboard KPI, punteggi prove, esportazione |

**Catena di avvio:**
[`main.py`](../main.py) → [`src/ui/main_gui.py`](../src/ui/main_gui.py) → [`MainWindowPro`](../src/ui/main_gui_pro.py)

---

## 3. Mappa del Repository

```
Dynamic_Target_Suite/
├── main.py                          ← Entrypoint dell'applicazione
├── src/
│   ├── models/                      ← Modelli fisici (3/7/14-DOF, pneumatici, powertrain)
│   ├── mdo/                         ← Motori di ottimizzazione e simulazione prove
│   ├── core/                        ← Thread fisica real-time, profili stradali, OpenMDAO
│   ├── ui/                          ← GUI PyQt5 (Pro + Classic), worker, dialoghi
│   └── utils/                       ← Gestore tracciati, mappe DNA, sensitività, digitizer
├── data/
│   ├── tire_tir/                    ← File .tir Pacejka (Hoosier 43075, SC24)
│   ├── tracks/                      ← Coordinate circuiti (.csv)
│   ├── setups/                      ← Setup veicolo esportati (.json)
│   └── motor_curve/                 ← Mappe efficienza EMRAX 188HV
├── tests/                           ← 218 test unitari (pytest / unittest)
├── docs/                            ← Guida sviluppatori, showcase, note matematiche
└── MDO_Target_Suite.spec            ← Spec di build PyInstaller
```

---

## 4. Simulazioni da Mostrare

### 4.1 Diagramma YMD & Handling

**Cosa mostra:** L'intera finestra di stabilità del veicolo — momento di imbardata *N* vs. accelerazione laterale *ay* a velocità fissa, con margine di controllo, margine di stabilità e linee di riferimento Ackermann sovrapposte.

**Come avviarlo:** GUI → Workspace Analisi → tab *Handling / YMD* → imposta velocità + pneumatico → *Run Sweep*.

**File chiave:** [`src/mdo/mdo_target_optimization.py`](../src/mdo/mdo_target_optimization.py) — `generate_handling_sweep_classic()`

**Perché è impressionante:**
- Sweep Pacejka non-lineare a 4 ruote: angoli di deriva anteriori/posteriori iterati all'equilibrio per ogni punto (β, δ)
- Margine di controllo CM = ∂N/∂δ e margine di stabilità SM = −∂N/∂β estratti e tracciati automaticamente
- Archi iso-raggio (Skidpad 9,125 m, curva Autocross) sovrapposti per contesto diretto di punteggio FSAE
- Tracking post-picco tramite continuazione pseudo-lunghezza d'arco (nessuna perdita di soluzione oltre il picco laterale)

**Cosa evidenziare nell'output:** La divergenza delle curve *N–ay* dall'Ackermann ideale, il punto neutro e il passaggio da sottosterzo a sovrasterzo.

---

### 4.2 Direct Collocation — Traiettoria Ottimale

**Cosa mostra:** La traiettoria a tempo minimo e il profilo di velocità per qualsiasi circuito FSAE, risolti come problema di controllo ottimo continuo.

**Come avviarlo:** GUI → Workspace MDO → *Direct Collocation* → seleziona tracciato → *Solve*.

**File chiave:** [`src/mdo/mdo_direct_collocation.py`](../src/mdo/mdo_direct_collocation.py) — `solve_direct_collocation_racing_line()`

**Perché è impressionante:**
- Formulato in coordinate curvilinee (avanzamento *s*, offset laterale *n*, errore di direzione *ξ*)
- Vettore di stato scalato a O(1) — variabile energetica E = ½v² invece della velocità; evita problemi di scaling numerico in IPOPT
- Vincolo ellisse di attrito combinato con scaling per deportanza aerodinamica
- Vincolo potenza massima 80 kW (regolamento FSAE EV)
- Warm-start: il profilatore quasi-statico backward-forward fornisce a IPOPT un punto iniziale ammissibile → convergenza tipica in < 60 iterazioni

**Cosa evidenziare nell'output:** La mappa colorata della velocità sulla geometria del tracciato, le zone di frenata forzate dalla curvatura e il confronto col profilo quasi-statico.

---

### 4.3 Sweep di Sensitività 1D Transitoria

**Cosa mostra:** L'influenza di ogni parametro di assetto (camber, convergenza, rigidezza ARB, molla, ecc.) su 21 KPI transitori in una manovra di step steer.

**Come avviarlo:** GUI → Workspace Analisi → tab *Sensitività* → seleziona scenario + range parametro → *Run Sweep*.

**File chiave:** [`src/utils/transient_sensitivity.py`](../src/utils/transient_sensitivity.py) — `run_sensitivity_sweep_1d()`

**KPI acquisiti per sweep (21 totali):**

| Categoria | KPI |
|-----------|-----|
| Risposta imbardata | r_ss, r_picco, OS%, t90, Δt_imb |
| Dinamica laterale | ay_picco, Δt_ay, TB-Factor, r_dot_picco |
| Stabilità | Gr, Gay, t_assest, β_picco |
| Trasferimento di carico | LLTD_picco, LLTD_ss, ΔFz_asim |
| Telaio | RollGrad, φ_max, η_μ |
| Utilizzo pneumatico | Util_A_max, Util_P_max |

**Cosa evidenziare nell'output:** Il grafico tornado con il ranking di influenza dei parametri, e come il gradiente di sottosterzo *K* varia con il rapporto di rigidezza anteriore/posteriore.

---

### 4.4 Simulatore Interattivo 14-DOF

**Cosa mostra:** Guida in tempo reale con fisica completa delle sospensioni, forze pneumatiche a 4 ruote, camber, convergenza e torque vectoring — giocabile con gamepad o volante force-feedback.

**Come avviarlo:** GUI → Workspace Simulatore → *Avvia Simulatore Interattivo* → collega gamepad.

**File chiave:**
- [`src/ui/interactive_driver_14dof.py`](../src/ui/interactive_driver_14dof.py) — viewport OpenGL/PyQt5
- [`src/core/physics_thread.py`](../src/core/physics_thread.py) — `PhysicsThread14DOF` a 250–500 Hz
- [`src/models/vehicle_14dof.py`](../src/models/vehicle_14dof.py) — ODE a 28 stati

**Perché è impressionante:**
- L'ODE è compilata in una **libreria C nativa (.dll) tramite CasADi → C → MSVC** al primo avvio, poi memorizzata nella cache con un hash SHA-256 dei parametri fisici — gli avvii successivi usano la DLL cached con overhead quasi nullo
- Integrazione RK4 completa a 500 Hz in un `QThread` dedicato — la GUI non si blocca mai
- Il torque vectoring (controllore PI di yaw rate) gira nello stesso thread fisico
- Telemetria live: angoli di deriva, trasferimento di carico, camber, accelerazione laterale — trasmessi ai grafici PyQt5 a 60 Hz

**Cosa evidenziare nell'output:** L'ellisse di utilizzo pneumatico in tempo reale e come satura in sovrasterzo. La barra di trasferimento di carico live tra asse anteriore e posteriore.

---

### 4.5 Scoring Skidpad & Accelerazione

**Cosa mostra:** Simulazione completa delle prove FSAE Skidpad e Accelerazione da 75 m con applicazione immediata della formula di punteggio FS2026.

**Come avviarlo:** GUI → Workspace MDO → *Prove* → *Skidpad* / *Accelerazione*.

**File chiave:**
- [`src/mdo/mdo_skidpad.py`](../src/mdo/mdo_skidpad.py) — solutore Skidpad stazionario + transitorio
- [`src/mdo/mdo_acceleration.py`](../src/mdo/mdo_acceleration.py) — dinamica longitudinale con controllo trazione
- [`src/mdo/mdo_sensitivity.py`](../src/mdo/mdo_sensitivity.py) — costanti di punteggio FS2026 + benchmark ranking mondiale

**Perché è impressionante:**
- Lo Skidpad usa il percorso a figura-8 di raggio 9,125 m con sequenza di ingresso/uscita regolamentare
- L'Accelerazione integra la mappa di coppia del motore EMRAX 188HV contro resistenza al rotolamento, resistenza aerodinamica e limite di trazione pneumatica — restituisce il tempo previsto con la storia degli interventi TCS
- Formula FS2026 applicata immediatamente: punti su 75 (Skidpad) e su 100 (Accelerazione) con benchmark di ranking mondiale

---

### 4.6 Risposta in Frequenza 7-Post Rig

**Cosa mostra:** Analisi modale delle sospensioni con input prescitti multi-asse da shaker (heave, pitch, roll, warp), calcolando la variazione del carico al contatto (CPLV) e la trasmissibilità del corpo vettura.

**Come avviarlo:** GUI → Workspace MDO → tab *7-Post Rig* → configura sweep in frequenza → *Run*.

**File chiave:**
- [`src/mdo/mdo_seven_post_rig.py`](../src/mdo/mdo_seven_post_rig.py) — fisica del rig
- [`src/ui/ui_seven_post_rig.py`](../src/ui/ui_seven_post_rig.py) — UI del workspace dedicato

**Perché è impressionante:**
- Input modali disaccoppiati: Heave (tutti e 4 i montanti in fase), Pitch (anteriore vs. posteriore), Roll (sinistra vs. destra), Warp (coppie diagonali)
- Restituisce frequenze naturali, rapporti di smorzamento, CPLV per angolo a ogni frequenza
- Correlazione diretta con le mappe ammortizzatori Öhlins TTX25 ([`src/models/damper_model_ttx25.py`](../src/models/damper_model_ttx25.py)) — mostra se le tarature scelte producono un isolamento ottimale

---

### 4.7 Mappe DNA del Veicolo (n1–n6)

**Cosa mostra:** Sei mappe di handling normalizzate (n1–n6) che caratterizzano il DNA non-lineare del veicolo (sottosterzo–sovrasterzo) in funzione di velocità, accelerazione laterale e carico sulle ruote.

**Come avviarlo:** GUI → Workspace Analisi → tab *Mappe DNA* → *Calcola n1–n6*.

**File chiave:** [`src/utils/vehicle_dna_maps.py`](../src/utils/vehicle_dna_maps.py)

**Perché è impressionante:**
- Mappe derivate dalla teoria non-lineare di handling veicolo — inclusa la relazione fondamentale ρ_y = −K/L
- Ogni mappa normalizzata a [−1, +1] per confronto diretto tra configurazioni diverse
- Continuazione pseudo-lunghezza d'arco per tracciare i rami oltre il regime limite senza perdere la soluzione

---

## 5. Sezioni di Codice Più Significative

### 5.1 Pipeline Compilazione ODE 14-DOF con CasADi

**File:** [`src/models/vehicle_14dof.py`](../src/models/vehicle_14dof.py)

Il componente tecnicamente più impegnativo del progetto. L'ODE del veicolo a 28 stati è scritta nel framework **simbolico** di CasADi (`casadi.SX`), il che consente:

1. **Differenziazione automatica** — Jacobiani esatti per IPOPT, nessuna differenza finita
2. **Generazione di codice C** — il grafo simbolico viene emesso come sorgente C tramite `casadi.CodeGenerator`
3. **Compilazione a DLL nativa** — il C viene compilato con MSVC/`cl.exe` in una `.dll` che gira a velocità quasi-C dentro Python tramite `ctypes`
4. **Cache basata su hash** — la DLL è memorizzata in `.cache/c_build/ode_14dof_<sha256>.dll` con chiave su tutti i parametri fisici; un cambio di parametri attiva automaticamente la ricompilazione

```python
# Pipeline concettuale (vedi vehicle_14dof.py per l'implementazione completa)
x    = ca.SX.sym('x', 28)  # vettore di stato (posizioni, velocità, stati pneumatico)
u    = ca.SX.sym('u',  4)  # ingressi (sterzo, acceleratore, freno, coppia TV)
xdot = vehicle_14dof_rhs(x, u, params)  # albero di espressioni simboliche

ode_fn = ca.Function('ode', [x, u], [xdot])
gen = ca.CodeGenerator('ode_14dof.c')
gen.add(ode_fn)
gen.generate()   # → ode_14dof.c
# poi: cl.exe → ode_14dof.dll  (nativa, cache in .cache/c_build/)
```

> [!NOTE]
> L'ODE è **sempre** scritta con primitive CasADi smooth. Nessun `if/else` Python su variabili simboliche — usare `ca.if_else`, `ca.tanh`, `ca.fmax` e regolarizzazioni smooth sqrt(x² + ε²) per le operazioni di modulo.

---

### 5.2 Modello Pneumatico Pacejka MF6.1 — Compatibile CasADi

**File:** [`src/models/tire_model_mf61.py`](../src/models/tire_model_mf61.py)  
**Dati:** [`src/models/params_tire.py`](../src/models/params_tire.py) + file `.tir` in [`data/tire_tir/`](../data/tire_tir/)

La formulazione MF6.1 completa (slittamento combinato, effetti camber, coefficienti normalizzati) implementata in matematica simbolica CasADi, così da poter essere:
- Incorporata nell'ODE 14-DOF (per esecuzione in DLL compilata)
- Differenziata esattamente per IPOPT nella direct collocation
- Valutata in NumPy puro per analisi offline veloci

Non-linearità chiave gestite in modo smooth:
- Saturazione dello slittamento: limite μ·Fz approssimato tramite sagomatura `tanh`
- Slittamento combinato: i fattori di ponderazione Brush/MF G_xα, G_yκ mantengono il corretto accoppiamento
- Mirroring TYRESIDE: un unico set di coefficienti si adatta a ruote sinistra e destra negando il camber e invertendo il segno della velocità laterale

> [!IMPORTANT]
> **Convenzione TYRESIDE:** I file `.tir` di questo progetto usano il lato `RIGHT` come riferimento (Hoosier 43075). Per le ruote di sinistra, il camber viene negato (γ_sin = −γ_raw) e il segno della velocità laterale viene invertito. **Non modificare questa logica di mirroring** senza rivalidare tutti i casi di test.

---

### 5.3 VehicleParams — Unica Fonte di Verità

**File:** [`src/models/params_vehicle.py`](../src/models/params_vehicle.py)

Ogni proprietà fisica e geometrica della vettura risiede nella dataclass `VehicleParams` — massa, passo, carreggiate, inerzie, geometria delle sospensioni, coefficienti aerodinamici, parametri del powertrain.

```python
from src.models.params_vehicle import VehicleParams, nominal_fsae_params

p = nominal_fsae_params()   # configurazione nominale FSAE E18
p.mass_total = 230.0        # modifica qualsiasi campo
```

**Regola:** Nessun modulo può cablare un valore numerico come `1.535` (passo) o `230.0` (massa). Tutte le costanti provengono da un'istanza `VehicleParams`. L'hash SHA-256 usato per la cache delle DLL è calcolato da questa dataclass — qualsiasi cambio di parametro attiva automaticamente la ricompilazione dell'ODE.

---

### 5.4 Ottimizzatore MDO Globale — Multi-Start SLSQP

**File:** [`src/mdo/mdo_global_optimizer.py`](../src/mdo/mdo_global_optimizer.py)

L'ottimizzatore di primo livello avvolge `scipy.optimize.minimize(method='SLSQP')` con:
- **Campionamento Latin Hypercube multi-start** — esplora lo spazio dei parametri prima della convergenza locale
- **Derivate di stabilità** calcolate dai gradienti YMD — ∂N/∂δ, ∂N/∂β usati come vincoli
- **Gradiente di sottosterzo standard** K = −L·ρ_y come vincolo soft con banda di tolleranza
- **OpenMDAO** per il coupling disciplinare quando carichi sulle ruote e aerodinamica interagiscono in modo non banale

La funzione obiettivo è una somma pesata dei punteggi delle prove (Skidpad, Accelerazione, tempo sul giro Endurance) calcolati dai rispettivi moduli MDO.

---

### 5.5 Worker QThread — Pattern di Calcolo Asincrono

**File:** [`src/ui/gui_workers.py`](../src/ui/gui_workers.py)

Ogni calcolo pesante è racchiuso in un worker basato su `QThread` che:
1. Riceve i parametri tramite segnali Python
2. Esegue il calcolo bloccante (IPOPT, SLSQP, integrazione ODE) in un **thread separato**
3. Emette `progress(int, str)` ogni N iterazioni per barre di progresso live
4. Emette `finished(result_dict)` al completamento
5. Emette `error(str)` in caso di eccezione — non crasha mai la GUI

```python
# Pattern usato ovunque nel progetto
class DirectCollocationWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def run(self):
        try:
            result = solve_direct_collocation_racing_line(
                self.track, self.params,
                progress_cb=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

> [!CAUTION]
> **Non chiamare mai** funzioni `mdo_*`, integrazione ODE o ottimizzatori direttamente da uno slot collegato a un pulsante GUI. Usare sempre un worker. Bloccare il thread principale congela il ciclo di eventi Qt e rende la finestra non responsiva.

---

## 6. Convenzioni Critiche e Insidie

### Ambiente Python
```powershell
# Corretto
.\venv\Scripts\python.exe main.py
.\venv\Scripts\python.exe -m pytest tests/

# Sbagliato — mancano CasADi, OpenMDAO, PyQt5
python main.py
```

### Convenzione di Segno del Gradiente di Sottosterzo
Il gradiente di sottosterzo è definito come:

    K = −L · ρ_y     dove  ρ_y = ∂ρ/∂ay

- **Sottosterzo:** K > 0 (K positivo, ρ_y negativo)
- **Sovrasterzo:** K < 0
- **Sterzo neutro:** K = 0
- **Condizione di stabilità:** ρ_y < 1/u²

> [!WARNING]
> **Non invertire mai** il segno di ρ_y o K. Diversi testi usano la convenzione opposta. L'intero sistema di mappe DNA e diagrammi di handling adotta questa convenzione — un'inversione produrrebbe valutazioni di stabilità errate.

### Variabili Simboliche CasADi
```python
# Corretto — primitive smooth CasADi
abs_v   = ca.sqrt(v**2 + 1e-6)                      # |v| regolarizzato
clamped = ca.fmax(0.0, ca.fmin(1.0, x))             # clamp [0, 1]
sign_v  = ca.tanh(v / 0.01)                         # segno smooth

# Sbagliato — flusso di controllo Python su valori simbolici
if v > 0:      # genera un'eccezione CasADi o si valuta in modo scorretto
    ...
abs_v = abs(v) # abs() Python non delega a CasADi
```

### Ricerca dei File Pneumatico
`TireParams._resolve_path()` cerca nell'ordine:
1. Percorso esatto come fornito
2. `data/tire_tir/<nome_file>`
3. `tire_tir/<nome_file>` (layout legacy)
4. Directory corrente, radice del repo, directory del modulo sorgente

Se un file `.tir` non viene trovato, il costruttore lancia `FileNotFoundError` con l'elenco completo dei percorsi cercati — controllare quella lista prima di segnalare un file mancante.

### Posizione Cache DLL
Le DLL ODE compilate finiscono in `.cache/c_build/`. Se la cache diventa troppo grande o una DLL risulta corrotta:
```powershell
Remove-Item .cache\c_build\*.dll -Force
# La prossima esecuzione ricompilerà automaticamente (~10–15 s al primo avvio)
```

### File di Risorse openmdao
Quando si esegue dall'**EXE compilato**, `openmdao` deve esistere come directory grezza in `_internal/` — non compressa nell'archivio PYZ. Questo è già configurato in [`MDO_Target_Suite.spec`](../MDO_Target_Suite.spec). Se appare `FileNotFoundError: unit_library.ini`, lo spec ha perso la configurazione `exclude_module` / `collect_all('openmdao')`.

### TYRESIDE nei File `.tir`
```
TYRESIDE = 'RIGHT'   ← riferimento Hoosier 43075
```
La fisica della ruota sinistra è calcolata negando il camber e invertendo il segno della velocità laterale **automaticamente** dentro `tire_model_mf61.py` tramite un flag `is_left_wheel`. Non pre-negare i valori prima di passarli al modello.

---

## 7. Suite di Test — Come Verificare che Tutto Funzioni

La suite di test conta **218 test** che coprono tutti i modelli fisici, le prove e le utility di analisi.

```powershell
# Esegui tutti i test (~7 minuti — inclusa compilazione ODE)
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"

# Solo i test di modello veloci
.\venv\Scripts\python.exe -m pytest tests/test_tire_mf61.py tests/test_3dof.py tests/test_damper_ttx25.py -v

# Pipeline CI unificata (test + GUI headless + health check)
.\venv\Scripts\python.exe scripts/ci_pipeline.py
```

**File di test principali da conoscere:**

| File di test | Cosa valida |
|-------------|-------------|
| `tests/test_14dof.py` | ODE 14-DOF: propagazione stato, cache DLL, effetti camber/convergenza |
| `tests/test_tire_mf61.py` | MF6.1: slittamento combinato, rigidezza camber, simmetria TYRESIDE |
| `tests/test_mmm_hd_verification.py` | Diagramma YMD / Handling: segni CM/SM, punto neutro |
| `tests/test_direct_collocation_smoke.py` | Ottimizzatore DC: convergenza IPOPT, soddisfacimento vincoli |
| `tests/test_transient_sensitivity.py` | Tutti i 21 KPI: segno, grandezza, controlli di monotonicità |
| `tests/test_vehicle_dna_maps.py` | Mappe DNA n1–n6: convenzione di segno del gradiente |
| `tests/test_dynamic_events.py` | Step Steer, DLC, Skidpad: limiti di sanità fisica |
| `tests/test_powertrain_electric.py` | Mappa coppia EMRAX, rigenerazione, allocazione TV |
| `tests/test_acceleration_and_diamond.py` | Tempo prova 75 m, mappa Diamond ARB |

---

## 8. Eseguibile Standalone

Per le demo senza ambiente Python installato:

```powershell
# Build (dalla radice del repo, richiede ~5–10 minuti)
.\venv\Scripts\pyinstaller.exe --noconfirm MDO_Target_Suite.spec

# Avvio
.\dist\MDO_Target_Suite\MDO_Target_Suite.exe
```

L'EXE include: PyQt5, CasADi, OpenMDAO, Matplotlib, SciPy, NumPy, Pygame, Pandas.  
I file dati pneumatico (`data/tire_tir/`) vengono copiati automaticamente in `_internal/tire_tir/` durante la build.

> [!TIP]
> Per la distribuzione, spedire **sempre** l'intera cartella `dist/MDO_Target_Suite/` — il solo eseguibile senza `_internal/` si arresta immediatamente all'avvio.

---

*Guida mantenuta dal team Vehicle Dynamics E18. Ultimo aggiornamento: ottobre 2026.*  
*Per le derivazioni matematiche, vedi [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md). Per le regole di contribuzione, vedi [`AGENTS.md`](AGENTS.md) e [`GEMINI.md`](GEMINI.md).*
