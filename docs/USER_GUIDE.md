# Manuale Utente & Guida di Studio — Dynamic Target Suite
**Autore:** Dynamic Target Suite Engineering Team  
**Versione:** 2.0 (Formula SAE / Formula Student & Race Car Dynamics)  
**Standard e Fondamenti di Riferimento:**
- SAE J670e / ISO 4138 (Vehicle Dynamics Terminology & Steady-State Handling).
- Pacejka Magic Formula Tire Representation (MF6.1).
- Non-linear Multibody & Stability Derivatives Formulation.
- Race Car Dynamics & Suspension Tuning Guidelines.

---

## Indice Generale
1. [Introduzione & Architettura della Suite](#1-introduzione--architettura-della-suite)
2. [Guida ai Modelli Fisici di Veicolo (3-DOF, 7-DOF, 14-DOF)](#2-guida-ai-modelli-fisici-di-veicolo)
3. [Ottimizzazione Globale MDO & Speed Sweep](#3-ottimizzazione-globale-mdo--speed-sweep)
4. [Studio & Interpretazione dei Diagrammi di Handling](#4-studio--interpretazione-dei-diagrammi-di-handling)
   - [4.1 Handling Diagram Cartesiano (Curvatura & Sottosterzo)](#41-handling-diagram-cartesiano)
   - [4.2 MAPs (Map of Achievable Performance)](#42-maps)
   - [4.3 Mappe DNA del Veicolo & Stabilità Modale (n1 ... n6)](#43-mappe-dna-del-veicolo)
   - [4.4 Moment Method & Yaw Moment Diagram (YMD)](#44-moment-method-e-yaw-moment-diagram)
   - [4.5 G-G-V 3D Performance Bowl](#45-g-g-v-3d-performance-bowl)
5. [Strumenti di Setup & Principi Fondamentali di Assetto](#5-strumenti-di-setup--principi-fondamentali-di-assetto)
   - [5.1 Sforzo Pilota & Coppia al Volante](#51-sforzo-pilota--coppia-al-volante)
   - [5.2 Dimensionamento Molle & Barre Antirollio ARB](#52-dimensionamento-molle--barre-antirollio-arb)
   - [5.3 Matrice & Radar 4-Rapporti Fondamentali](#53-matrice--radar-4-rapporti-fondamentali)
6. [Sospensioni Reali: Molle, Tamponi di Fine Corsa (Bump Stops) & Altezze](#6-sospensioni-reali-molle-tamponi-e-altezze)
7. [Mappe Cinematiche Non-Lineari & Curve Digitizer](#7-mappe-cinematiche-non-lineari--curve-digitizer)
8. [Analisi di Sensitività Transitoria (1D & 2D Heatmaps)](#8-analisi-di-sensitività-transitoria)
9. [Simulatore di Guida 14-DOF Real-Time & Direct Collocation](#9-simulatore-di-guida-14-dof-real-time)
10. [Vademecum di Setup in Pista (Track Engineering Cheatsheet)](#10-vademecum-di-setup-in-pista)

---

## 1. Introduzione & Architettura della Suite

La **Dynamic Target Suite** è un ambiente integrato di progettazione, ottimizzazione multidisciplinare (MDO), simulazione e validazione virtuale per vetture da competizione (Formula SAE / Formula Student e prototipi sportivi).

### Obiettivo Fondamentale
Definire e sintetizzare i parametri geometrici, cinematici, aerodinamici e di assetto ottimali per massimizzare la prestazione in aderenza laterale e longitudinale (ay, ax), garantendo contemporaneamente:
1. **Stabilità direzionale intrinseca** a qualsiasi regime di velocità (rho_y < 1/u^2, N_beta > 0).
2. **Prevedibilità e linearità di risposta** per il pilota (controllo del gradiente di sottosterzo Kus > 0 e dello sforzo al volante < 5.0 Nm).
3. **Robustezza aerodinamica e rispetto dei fine corsa meccanici** (escursione reale degli ammortizzatori, ingaggio progressivo dei bump-stops e mantenimento del fondo vettura al di sopra del terreno).

---

## 2. Guida ai Modelli Fisici di Veicolo

La suite implementa tre livelli di fedeltà fisico-matematica, consentendo di passare rapidamente dal calcolo concettuale alla validazione transitoria a corpo rigido completo.

`
                    +-------------------------+
                    |  3-DOF Steady-State     |  -> Sintesi Rapida MDO & Handling Base
                    +-----------+-------------+
                                |
                    +-----------v-------------+
                    |  7-DOF Transient Trim   |  -> Dinamica Planare + Rollio + Powertrain
                    +-----------+-------------+
                                |
                    +-----------v-------------+
                    |  14-DOF Full Vehicle    |  -> Corsa Ruote, Bump Stops, Camber/Toe
                    +-------------------------+    Dinamici, Caster Jacking & Simulatore
`

### 2.1 Modello 3-DOF (Steady-State MDO)
- **Gradi di Libertà**: Velocità laterale vy, Yaw Rate r, derivati in regime stazionario (vy_dot = 0, r_dot = 0).
- **Uso ideale**: Ottimizzazione globale rapida dei parametri primari: passo L, carreggiate t_f, t_r, distribuzione dei pesi wd_f, ripartizione del trasferimento di carico laterale LLTD, bilanciamento aerodinamico aero_bal.

### 2.2 Modello 7-DOF (Transient Trim)
- **Gradi di Libertà (7)**: vx, vy, r, phi (rollio telaio), phi_dot (velocità di rollio), w_motore, delta_steer.
- **Caratteristiche**: Include l inerzia di rollio Ix, la rigidezza delle barre antirollio (K_phi_f, K_phi_r), lo smorzamento di rollio e la curva di erogazione del motore elettrico con limitatore da regolamento FSAE (80 kW).

### 2.3 Modello 14-DOF (Full Vehicle & Kinematics)
- **Gradi di Libertà (14)**:
  - *Moto Corpo Vettura (6)*: vx, vy, r, rollio phi, phi_dot, beccheggio theta, theta_dot.
  - *Escursione Sospensioni (4)*: Corse verticali ruota z_fl, z_fr, z_rl, z_rr con molle reali, fine corsa fisici e tamponi progressivi.
  - *Dinamica Ruote (4)*: Velocità angolari w_fl, w_fr, w_rl, w_rr con modello combinato Pacejka MF6.1.
- **Cinematica Avanzata**:
  - Camber Gain e Toe Gain (Bump-Steer) non lineari dipendenti dall escursione z.
  - Altezza e migrazione laterale dei Centri di Rollio h_rc(z), y_rc(Delta z).
  - Caster Jacking & KPI Jacking: ridistribuzione del carico diagonale con lo sterzo.
  - Ripartizione di frenata idraulica (brake_bias) e Torque Vectoring elettronico.

---

## 3. Ottimizzazione Globale MDO & Speed Sweep

L ottimizzazione multidisciplinare (MDO) permette di identificare il punto di massimo rendimento sul veicolo globale.

### 3.1 Formulazione del Problema
Minimizzazione di J(x) = -(max ay(x, vx)) + w_pen * P_understeer(x) + w_stab * P_highspeed(x) sotto vincoli geometrici ed escursioni definite dall utente.

### 3.2 Penalità di Sottosterzo e Stabilità
Per evitare che l ottimizzatore sfrutti matematicamente condizioni sovrasterzanti al limite (instabili e non guidabili da un pilota umano), il software applica:
1. **Penalità Quadratica di Sottosterzo**:
   P_understeer = max(0, Kus_min - Kus(x))^2.
   Garantisce che il gradiente di sottosterzo calcolato rimanga strettamente al di sopra del target desiderato (es. +1.0 deg/g).
2. **Regola dei 4 Rapporti Fondamentali**:
   Il bilanciamento LLTD deve essere superiore all Aero Balance di almeno il 2% (LLTD >= Aero + 0.02) per evitare sovrasterzo asintotico nei curvoni veloci.

### 3.3 Speed Sweep Parallelo
Viene eseguito uno sweep a velocità crescente (es. da 10 m/s a 28 m/s a passi di 2 m/s) sfruttando il parallelismo multi-core CPU. Il grafico risultante illustra come deve evolvere l assetto (LLTD, camber, altezze e rigidezze) all aumentare del carico verticale generato dalle ali.

---

## 4. Studio & Interpretazione dei Diagrammi di Handling

### 4.1 Handling Diagram Cartesiano
**Riferimento:** Formulazione classica dell'Handling Diagram (Pacejka MF6.1 / Standard SAE).

Il diagramma di handling traccia la differenza degli angoli di deriva (alpha_f - alpha_r) = (delta - L/R) rispetto all accelerazione laterale ay:
- **Pendenza Positiva (Kus > 0):** Veicolo **sottosterzante**. All aumentare di ay il pilota deve incrementare l angolo di sterzo più di quanto farebbe a bassissima velocità (cinematica di Ackermann L/R).
- **Pendenza Verticale (Kus = 0):** Veicolo **neutro**.
- **Pendenza Negativa (Kus < 0):** Veicolo **sovrasterzante**. Per mantenere la traiettoria, il pilota deve controsterzare o ridurre l angolo al volante.
- **Asintoto Orizzontale (ay_max):** Limite fisico di grip della vettura.

### 4.2 MAPs (Map of Achievable Performance)
**Riferimento:** Rappresentazione a quadranti delle prestazioni stazionarie limite.

Le MAPs scompongono l intero dominio operativo del veicolo nei quattro quadranti fondamentali:
1. **(delta, u) — Input Region**: Mostra per ciascuna velocità quale angolo di sterzo alle ruote delta è fisicamente attuabile prima della perdita di aderenza.
2. **(u, ay) — Performance Envelope**: Andamento della massima accelerazione laterale in funzione della velocità di avanzamento u.
3. **(delta, ay) — Control Characteristic**: Risposta in ay per ogni grado di sterzo. La pendenza indica la sensibilità o gain di sterzo.
4. **(rho, beta) & (delta, rho) — Output/Curvature Regions**: Rappresentano la curvatura di traiettoria rho = 1/R generata e l angolo di assetto beta.

#### Standard Matematico del Gradiente:
rho_y = d(rho)/d(ay) = -K/L
- **Condizione di Stabilità Direzionale:** rho_y < 1/u^2. Se rho_y >= 1/u^2, il veicolo supera la velocità critica v_crit = sqrt(-L/K) e diventa intrinsecamente instabile.

### 4.3 Mappe DNA del Veicolo & Stabilità Modale (n1 ... n6)
Le mappe DNA analizzano le **derivate parziali dimensionali** della dinamica del veicolo:
- n1 ... n6: Sensibilità dei carichi normali, forze laterali e momenti di imbardata rispetto a beta, r, delta.
- **Locus dei Poli / Modi Oscillatori:** Visualizza la frequenza naturale fn [Hz] e il coefficiente di smorzamento zeta del modo d imbardata (Yaw Mode) per confermare l assenza di oscillazioni o instabilità divergenti.

### 4.4 Moment Method & Yaw Moment Diagram (YMD)
**Riferimento:** Metodo del Momento d'Imbardata / Diagramma Mz-ay.

Il diagramma YMD mappa le forze nel piano (ay, Mz) a velocità fissa vx:
- **Linea Mz = 0 (Trim Stazionario):** Punti di equilibrio stazionario in curva senza imbardata residua.
- **Spaziatura Linee Iso-delta (dMz/ddelta):** **Autorità dello sterzo**. Una spaziatura ampia garantisce reattività e inserimento immediato.
- **Pendenza Linee Iso-beta (dMz/dbeta):** **Stabilità direzionale statica**. Pendenza negativa -> veicolo stabile; pendenza positiva -> veicolo instabile a comando bloccato.

### 4.5 G-G-V 3D Performance Bowl
Rappresenta l ellisse di attrito 3D (ax, ay, vx):
- A bassa velocità (vx < 12 m/s), l inviluppo è simmetrico e dominato dall aderenza meccanica degli pneumatici.
- Ad alta velocità (vx > 20 m/s), la deportanza allarga l inviluppo laterale a oltre 2.0 - 2.5 g, mentre la potenza limitata del motore e la resistenza all avanzamento stringono la zona in trazione positiva (ax > 0).

---

## 5. Strumenti di Setup & Principi Fondamentali di Assetto

### 5.1 Sforzo Pilota & Coppia al Volante
- **Trail Totale:** tau_tot = t_p + t_m (somma di trail pneumatico e meccanico).
- **Schwarzenegger Limit:** La coppia continua al volante non deve superare **5.0 Nm** per evitare l affaticamento del pilota nelle gare lunghe di Endurance (22 km).

### 5.2 Dimensionamento Molle & Barre Antirollio ARB
- **Flat Ride Principle:** La frequenza naturale del retrotreno deve essere leggermente superiore a quella dell avantreno (es. fn,f = 3.0 Hz, fn,r = 3.3 Hz) in modo che il retrotreno insegua l avantreno riducendo il beccheggio sui dossi.
- **Dimensionamento Tubi Cavi ARB:** Calcolo analitico del diametro esterno Do del tubo (acciaio G=79.3 GPa o titanio G=45 GPa) per ottenere la rigidezza di rollio residua necessaria a soddisfare il target LLTD.

### 5.3 Matrice & Radar 4-Rapporti Fondamentali
La dinamica di un auto da corsa è governata da 4 rapporti fondamentali:
1. **Weight Distribution (wd_f):** Bilanciamento delle masse statiche.
2. **Lateral Load Transfer Distribution (LLTD):** Ripartizione elastica del rollio anteriore/posteriore.
3. **Aerodynamic Balance (Aero Bal):** Centro di pressione dei carichi deportanti.
4. **Track Width Ratio (t_f / t_r):** Rapporto tra carreggiata anteriore e posteriore.

**Regola Aurea di Stabilità ad Alta Velocità:**
LLTD >= Aero Balance + 2.0%
Se il bilanciamento aerodinamico supera l LLTD, l auto diventerà instabile nei curvoni ad alta velocità all aumentare dell efficienza delle ali.

---

## 6. Sospensioni Reali: Molle, Tamponi di Fine Corsa (Bump Stops) & Altezze

Nel modello 14-DOF e nel tab di dimensionamento, la suite modella il comportamento fisico reale della sospensione:

1. **Corsa Libera & Ingaggio Bump-Stop:**
   Finché z <= z_gap (es. 20 mm di compressione), agisce la sola molla installata Kw = K_spring * MR^2.
   Quando z > z_gap, si somma la rigidezza progressiva del tampone in elastomero:
   F_susp(z) = Kw * z + K_bs * (z - z_gap) + 50 * K_bs * (z - z_gap)^2.
2. **Altezza Dinamica dal Suolo (Ground Clearance):**
   h_ground(t) = h_static - z(t) - Delta z_tire.
   Se h_ground <= 0, scatta l allarme di **Bottoming-Out / Contatto Suolo**, indicando la necessità di aumentare il precarico o la rigidezza molla.

---

## 7. Mappe Cinematiche Non-Lineari & Curve Digitizer

La suite consente di digitalizzare curve reali da software di cinematica (Lotus Shark, Adams Car, SolidWorks, SusProg):
1. **Avvio Digitizer:** Menu Tools -> Curve Digitizer.
2. **Caricamento Immagine:** Selezionare lo screenshot del grafico (es. Camber vs Wheel Travel).
3. **Calibrazione Assi:** Cliccare sull origine (0,0), estremo asse X e asse Y specificando le unità di misura.
4. **Digitizzazione Punti & Fitting:** Cliccare lungo la curva. L algoritmo esegue una regressione polinomiale di 3° grado:
   y(z) = a3 * z^3 + a2 * z^2 + a1 * z + a0.
5. **Esportazione JSON:** Il file salvato converte automaticamente le grandezze in unità standard SI (metri per l escursione ruota).
6. **Caricamento in GUI:** Nel Sub-Tab 5 di Setup Tools, cliccare Carica JSON (Digitizer) per applicare immediatamente la curva al veicolo attivo.

---

## 8. Analisi di Sensitività Transitoria (1D & 2D Heatmaps)

Permette di studiare l effetto della variazione di uno o due parametri su manovre dinamiche standard (**Step Steer** o **Double Lane Change - DLC**):
- **Visualizzazione 1D:** Mostra le risposte temporali di imbardata r(t), rollio phi(t) e LLTD(t), con **Colorbar graduata** che mappa ciascun colore al valore numerico del parametro.
- **Visualizzazione 2D (Heatmap Bivariata):** Mappa l interazione di due parametri (es. altezza centro di rollio ant. q1 x post. q2) su un KPI obiettivo (es. LLTD di picco, tempo di risposta t_90, o overshoot di rollio).

---

## 9. Simulatore di Guida 14-DOF Real-Time & Direct Collocation

### 9.1 Simulatore Interattivo Gamepad / Tastiera
Avviabile da menu Tools -> 14-DOF Gamepad Driving Simulator o da pulsante rapido.
- Supporta Gamepad (Xbox, PlayStation, Logitech Wheel con Force Feedback) e Tastiera.
- Risolve l ODE a 14 gradi di libertà a **100 Hz** in tempo reale con visualizzazione 3D wireframe del telaio, ruote sterzanti e telemetria live su 4 quadranti.

### 9.2 Direct Collocation (Racing Line Ottima)
Avviabile da menu Tools -> Direct Collocation (Racing Line Optimization).
- Risolve il problema di controllo ottimo del tempo di giro (**Minimum Lap Time**) discretizzando il tracciato con polinomi ortogonali di Radau/Gauss.
- Ottimizza simultaneamente la traiettoria all interno dei cordoli della pista e i profili di sterzo, accelerazione e frenata, confrontandoli con il Backward Profiler classico.

---

## 10. Vademecum di Setup in Pista (Track Engineering Cheatsheet)

| Sintomo Dinamico Riscontrato | Causa Fisica Primaria | Intervento di Assetto Consigliato |
| :--- | :--- | :--- |
| **Sottosterzo in Inserimento** (Entry Understeer) | Avantreno rigido o insufficiente trasferimento di carico iniziale | - Ammorbidire barra antirollio anteriore (ARB F)<br/>- Ridurre smorzamento in compressione (Bump) ant.<br/>- Incrementare convergenza divergente ant. (Toe-Out) |
| **Sovrasterzo in Inserimento** (Entry Oversteer / Snap) | Eccessivo trasferimento di carico posteriore in frenata | - Spostare ripartizione di frenata in avanti (brake_bias)<br/>- Aumentare smorzamento in estensione (Rebound) post.<br/>- Aumentare convergenza convergente post. (Toe-In) |
| **Sottosterzo a Centro Curva** (Mid-Corner Push) | LLTD anteriore eccessivo o insufficiente camber dinamico | - Ridurre diametro/rigidezza ARB anteriore<br/>- Aumentare camber negativo statico anteriore<br/>- Alzare centro di rollio posteriore (q2) |
| **Sovrasterzo a Centro Curva** (Mid-Corner Loose) | LLTD posteriore eccessivo o retrotreno rigido | - Aumentare rigidezza ARB anteriore<br/>- Ridurre rigidezza molle/ARB posteriori |
| **Sottosterzo in Uscita** (Exit Understeer on Power) | Trasferimento di carico longitudinale che scarica l avantreno | - Ammorbidire barra ARB anteriore<br/>- Aumentare smorzamento Rebound anteriore e Bump posteriore |
| **Sovrasterzo in Uscita** (Exit Oversteer / Power Oversteer) | Saturazione di trazione combinata posteriore | - Ridurre precarico differenziale o Torque Vectoring<br/>- Ammorbidire molle posteriori<br/>- Aumentare ala posteriore (Aero Balance) |
| **Eccessivo Sforzo al Volante** (T_sw > 5.0 Nm) | Trail meccanico/pneumatico elevato o scrub radius eccessivo | - Ridurre angolo di Caster<br/>- Ridurre braccio a terra trasversale (Scrub Radius)<br/>- Aumentare rapporto di sterzo (GR) |
| **Fondo Vettura che Tocca (Scraping)** | Corse a tampone esaurite o molle troppo morbide | - Aumentare altezze statiche da terra (h_ride)<br/>- Ridurre gap tampone o aumentare rigidezza molle |

---

## 11. Rombo di Setting ARB & Bilanciamento Meccanico 14-DOF

Accessibile dalla scheda **"Rombo ARB 14-DOF"** (Workspace 2: Assetto & Componenti Sospensioni nella GUI Pro, o Sub-Tab 4 in Setup Tools nella GUI Classic):
- **Piano di Lavoro**:
  - Asse X: Rigidezza a rollio totale $K_{\phi,\text{tot}} = K_{\phi,f} + K_{\phi,r}$ [Nm/deg].
  - Asse Y: Ripartizione rigidezza anteriore $\text{Ratio}_f = \frac{K_{\phi,f}}{K_{\phi,\text{tot}}}$ [%].
- **Configurazione dei Vertici & Gestione Profili ("Salva con Nome...")**:
  - La tabella in alto a sinistra mostra i 4 vertici fisici del rombo ($V_1$ Soft-Soft, $V_2$ Stiff F / Soft R, $V_3$ Stiff-Stiff, $V_4$ Soft F / Stiff R). I valori possono essere modificati direttamente editando le celle numeriche.
  - **Profilo / Preset**: selettore a tendina con i profili memorizzati in `setups/arb_diamond/` (*ET18 FSAE Predefinito*, *Asciutto High Grip*, *Bagnato Rain*, *Ampio Range*).
  - **💾 Salva**: salva le modifiche direttamente sul profilo corrente.
  - **💾 Salva con Nome...**: salva la configurazione corrente in un nuovo file JSON con nome a scelta nella cartella dei preset.
  - **📂 Sfoglia...**: apre un file di configurazione rombo da qualsiasi directory.
  - **🗑 Elimina**: elimina un profilo personalizzato salvato in precedenza.
  - **📐 Calcola da ARB...**: apre una finestra di dialogo guidata per calcolare automaticamente i vertici a partire dai limiti dimensionali e torsionali min/max delle barre antirollio e dalle molle installate.
- **Funzioni di Analisi 14-DOF ad Esecuzione Asincrona (Zero Blocco UI)**:
  - Tutte le analisi girano in background su un worker thread dedicato, mostrando una **barra di avanzamento percentuale** e il pulsante **"✖ Annulla"** per fermare il calcolo in qualsiasi istante senza congelare la finestra.
  - **Analizza Vertici (14-DOF)**: computa la performance a regime (Skidpad $a_{y,\max}$, tempo sul giro, gradiente di rollio) e la risposta dinamica transitoria (Step Steer $t_{90}$, overshoot di imbardata, differenziale di utilizzazione attrito $\Delta \text{Util}$). Mostra il rombo con il setup attuale evidenziato, il radar chart e le risposte temporali.
  - **Mappa 2D Rombo (Contour)**: genera le mappe a curve di livello di $a_{y,\max}$ e del gradiente di sottosterzo $K_{us}$ standard (SAE / Pacejka) all'interno del rombo (calcolo ultrarapido ~4s), localizzando visivamente lo **Sweet Spot** (stella dorata) di massimo grip in pieno controllo di sottosterzo progressivo.
  - **Sweep 1D Bilanciamento**: varia con precisione la sola ripartizione di rollio anteriore a rigidezza totale costante (~0.3s).
  - **Applica Ottimo Consigliato al Setup Attivo**: applica immediatamente i valori calcolati ($K_{\phi,f}, K_{\phi,r}, \text{lltd}$) alla vettura attiva e aggiorna l'intero database parametri.

---
*Fine del Manuale Utente & Guida di Studio — Dynamic Target Suite*


