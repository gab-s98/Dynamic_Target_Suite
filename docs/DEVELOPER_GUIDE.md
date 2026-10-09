# Dynamic Target Suite — Guida per Programmatori & Architettura Software

Benvenuto nella documentazione tecnica ufficiale della **Dynamic Target Suite**.  
Questa guida è rivolta agli sviluppatori, ingegneri del veicolo e maintainer del progetto. Fornisce l'architettura completa del software, il data flow tra i moduli matematici e la GUI, la spiegazione dei modelli a 3, 7 e 14 gradi di libertà, i solutori di ottimizzazione globale, la Direct Collocation, il motore di sensitività transitoria, le Mappe del DNA del Veicolo e il simulatore di guida interattivo in tempo reale.

---

## 1. Architettura Generale del Software

La suite è sviluppata in **Python 3.12** e adotta un'architettura modulare ad alte prestazioni:
- **Front-End Grafico**: **PyQt5** con canvas **Matplotlib** interattivi (legende trascinabili, zoom continuo, colormap turbo, sub-tab di telemetria).
- **Motore Simbolico & Differenziazione Automatica**: **CasADi** (interfacciato con i solutori C/Fortran ad alte prestazioni **IPOPT** e **MUMPS**).
- **Calcolo Numerico & Statistico**: **NumPy**, **SciPy** (`optimize.minimize` SLSQP/Multi-Start, `spatial.ConvexHull`, `interpolate.interp1d`), **Pandas**.
- **Gestione Controller**: **Pygame Joystick API** e driver DirectInput con fallback su tastiera.

```
Dynamic_Target_Suite/
│
├── main.py                             # Entrypoint principale dell'applicazione
├── requirements.txt                    # Dipendenze Python
├── pytest.ini                          # Configurazione runner test
├── build_exe.bat                       # Script di compilazione PyInstaller
├── MDO_Target_Suite.spec               # Specfile PyInstaller
├── README.md & LICENSE                 # Documentazione di primo livello e licenza
│
├── src/                                # Package sorgente principale
│   ├── models/                         # [MODELLI FISICI, PNEUMATICI & PARAMETRI]
│   │   ├── params_vehicle.py           # SINGLE SOURCE OF TRUTH: Dataclass VehicleParams & catalogo setup
│   │   ├── params_tire.py              # Parser file .TIR (Pacejka MF6.1)
│   │   ├── tire_model_mf61.py          # Modello pneumatico non-lineare MF6.1 simbolico/vettoriale
│   │   ├── tire_degradation.py         # Modello termico/degradazione battistrada
│   │   ├── vehicle_3dof.py             # Modello 3-DOF Steady-State (CasADi Newton trim)
│   │   ├── vehicle_7dof.py             # Modello 7-DOF Transitorio (imbardata, rollio, 4 ruote)
│   │   ├── vehicle_14dof.py            # Modello 14-DOF Full Vehicle (sospensioni 3D, cinematica, bump stop)
│   │   ├── damper_model_ttx25.py       # Modello ammortizzatori Öhlins TTX25 MkII
│   │   ├── powertrain_electric.py      # Motore EMRAX 188HV, mappe efficienza & TV
│   │   ├── steering_dynamics.py        # Sforzo al volante e bracci a terra
│   │   └── vehicle_3d_models.py        # Mesh geometriche 3D telaio e ruote
│   │
│   ├── mdo/                            # [MDO & SOLUTORI NUMERICI]
│   │   ├── mdo_global_optimizer.py     # Ottimizzatore SLSQP multi-start & derivate di stabilità
│   │   ├── mdo_target_optimization.py  # Handling Diagram, MAPs (Achievable Performance), YMD
│   │   ├── mdo_direct_collocation.py   # Traiettoria ottima CasADi + IPOPT in coordinate curvilinee
│   │   ├── mdo_dynamic_events.py       # Simulazione eventi (Step Steer, DLC, Slalom, Endurance)
│   │   ├── mdo_skidpad.py              # Solutore Skidpad stazionario e transitorio
│   │   ├── mdo_acceleration.py         # Simulatore accelerazione longitudinale pura
│   │   ├── mdo_balance_range.py        # Analisi finestre di bilanciamento
│   │   ├── mdo_arb_diamond_balance.py  # Mappa Diamond barre antirollio
│   │   ├── mdo_seven_post_rig.py       # Core fisico 7-Post Rig shaker
│   │   ├── mdo_sensitivity.py          # Analisi sensitività globale
│   │   ├── mdo_uncertainty.py          # Propagazione incertezze
│   │   └── mdo_steady_state.py         # Valutazione KPI stazionari
│   │
│   ├── core/                           # [CORE ENGINE, EQUILIBRIO & SIMULAZIONE]
│   │   ├── simulator.py                # Wrapper unificato di simulazione
│   │   ├── om_vehicle_equilibrium.py   # Componente OpenMDAO equilibrio veicolo
│   │   ├── road_profile.py             # Generatore profili stradali (PSD, bump, sintetici)
│   │   ├── simulation_options.py       # Opzioni e tolleranze di simulazione
│   │   ├── gamepad_manager.py          # Gestione input controller e volanti FFB
│   │   ├── physics_thread.py           # Thread real-time physics a 250-500 Hz
│   │   └── vehicle_sim_controller.py   # Controller di simulazione interattivo
│   │
│   ├── ui/                             # [INTERFACCIA GRAFICA & SIMULATORE]
│   │   ├── main_gui.py                 # Entry point interfaccia grafica
│   │   ├── main_gui_pro.py             # UI Moderna: 5 Workspace, zero emoji, docking pulito
│   │   ├── main_gui_classic.py         # UI di riserva/backup classica
│   │   ├── interactive_driver_14dof.py # Simulatore di guida interattivo (OpenGL/PyQt5)
│   │   ├── gui_canvas.py               # Canvas Matplotlib riutilizzabile per PyQt5
│   │   ├── gui_dialogs.py              # Finestre modali e dialoghi di supporto
│   │   ├── gui_workers.py              # Worker QThread asincroni per calcolo non bloccante
│   │   ├── ui_theme_engine.py          # Gestore temi QSS e stile grafici
│   │   ├── ui_seven_post_rig.py        # UI Workspace per 7-Post Rig
│   │   └── legacy/                     # Re-export per compatibilità retroattiva
│   │
│   └── utils/                          # [TOOLSET DI SETUP & UTILITY]
│       ├── track_manager.py            # Gestore tracciati e superficie stradale 2D/3D
│       ├── curve_digitizer.py          # Tool estrazione curve camber/toe da immagini
│       ├── vehicle_dna_maps.py         # Mappe del DNA del Veicolo n1..n6 e stabilità modale
│       ├── optimized_params_manager.py # Gestore persistenza parametri ottimizzati
│       ├── suspension_sizing.py        # Dimensionamento molle, barre ARB e Radar 4-Rapporti
│       ├── transient_sensitivity.py    # Sweep 1D/2D con estrazione 21 KPI dinamici
│       └── logger.py                   # Logging strutturato
│
├── data/                               # [DATI, TRACCIATI E CONFIGURAZIONI]
│   ├── tire_tir/                       # File .tir ufficiali Pacejka MF6.1
│   ├── tire_datas/                     # Dati sperimentali banco pneumatici
│   ├── tracks/                         # Coordinate tracciati (.csv, .mat)
│   ├── setups/                         # Setup veicolo esportati (.json)
│   ├── motor_curve/                    # Curve caratteristiche motore
│   └── design_suspension/              # Rilievi cinematici sospensioni
│
├── docs/                               # [DOCUMENTAZIONE TECNICA]
│   ├── DEVELOPER_GUIDE.md              # Architettura approfondita e formule LaTeX
│   ├── md_equations/                   # Note matematiche di dettaglio
│   ├── Mathematica_Reference/          # Riferimenti analitici
│   └── documents/                      # Paper e documentazione tecnica
│
├── tests/                              # [SUITE TEST UNITARI & INTEGRAZIONE]
│   ├── test_*.py                       # Test unitari conformi a unittest/pytest
│   └── ...
│
├── scripts/                            # [AUTOMAZIONE & PIPELINE CI]
│   ├── ci_pipeline.py                  # Script unificato CI (Test + Headless GUI + Build)
│   └── ...
│
└── [RUNTIMES & CACHE - IGNORATI DA GIT]
    ├── .cache/c_build/                 # DLL compilate C/CasADi (`ode_14dof_*.dll`)
    ├── runs/                           # Output simulazioni e log OpenMDAO
    └── venv/                           # Ambiente virtuale locale
```

---

## 2. Gerarchia dei Modelli di Veicolo (Model Tiers)

La suite supporta 3 livelli di complessità fisica rigorosamente gerarchici, implementati nei package `src/models/` e selezionabili dinamicamente dall'interfaccia:

### Tier 1: Modello 3-DOF Steady-State (`src/models/vehicle_3dof.py`)
- **Stati**: $[\beta, r]$ (oppure $v_y, r$) a velocità $v_x$ imposta.
- **Utilizzo**: Calcolo istantaneo del trim stazionario, generatore di diagrammi di Handling, MAPs (Map of Achievable Performance) ed equilibrio rapido.
- **Risoluzione Numerica**: Solutore Newton-Raphson simbolico CasADi ad altissima convergenza radiale.
- **Equazioni di Equilibrio**:
  $$\Sigma F_x = 0 \iff F_{x,f}\cos\delta - F_{y,f}\sin\delta + F_{x,r} - F_{\text{drag}} = 0$$
  $$\Sigma F_y = m v_x r \iff F_{y,f}\cos\delta + F_{x,f}\sin\delta + F_{y,r} = m v_x r$$
  $$\Sigma M_z = 0 \iff a (F_{y,f}\cos\delta + F_{x,f}\sin\delta) - b F_{y,r} = 0$$
- **Distribuzione Carichi Verticali con Aerodinamica**:
  $$F_{z,f} = m g \frac{b}{L} + \frac{1}{2}\rho S C_z v_x^2 \cdot \text{bal}_{\text{aero}}$$
  $$F_{z,r} = m g \frac{a}{L} + \frac{1}{2}\rho S C_z v_x^2 \cdot (1 - \text{bal}_{\text{aero}})$$
  Il trasferimento di carico laterale dinamico è accoppiato con l'accelerazione centripeta $a_y = v_x r$ e con la ripartizione $\text{LLTD}$.

### Tier 2: Modello 7-DOF Transitorio (`src/models/vehicle_7dof.py`)
- **Stati (7)**: $[u, v, r, \omega_{fl}, \omega_{fr}, \omega_{rl}, \omega_{rr}]$.
- **Assetto di Rollio Accoppiato**: L'angolo di rollio $\phi(t)$ è ricavato dinamicamente dall'equilibrio dei momenti:
  $$(I_{xx} + m_s h_{\text{arm}}^2)\ddot{\phi} + C_\phi \dot{\phi} + K_\phi \phi = m_s a_y h_{\text{arm}}$$
- **Utilizzo**: Dinamica nel dominio del tempo, manovre transitorie con dinamica ruota e slittamento combinato (Step Steer, Skidpad transitorio).
- **Equazioni Dinamica di Cassa**:
  $$m (\dot{u} - v r) = \sum F_{x,i} - F_{\text{aero,drag}}$$
  $$m (\dot{v} + u r) = \sum F_{y,i}$$
  $$I_{zz} \dot{r} = a (F_{y,fl} + F_{y,fr}) - b (F_{y,rl} + F_{y,rr}) + \frac{t_f}{2}(F_{x,fr} - F_{x,fl}) + \frac{t_r}{2}(F_{x,rr} - F_{x,rl}) + M_{z,TV}$$
- **Equazioni Dinamiche Ruote (4)**:
  $$I_w \dot{\omega}_i = T_{\text{drive},i} - T_{\text{brake},i} - F_{x,i} R_w$$
  con supporto a differenziale aperto, differenziale a slittamento limitato (LSD) e Active Torque Vectoring.

### Tier 3: Modello 14-DOF Full Vehicle (`src/models/vehicle_14dof.py`)
- **Stati (28)**:
  - Moto planare cassa: $[v_x, v_y, r]$
  - Assetto 3D cassa: Rollio $[\phi, \dot{\phi}]$, Beccheggio $[\theta, \dot{\theta}]$, Heave cassa $[w_{\text{heave}}]$ (velocità verticale baricentro lungo ISO $Z+$)
  - 4 Masse non sospese al 2° ordine: corse sospensione $[z_{fl}, z_{fr}, z_{rl}, z_{rr}]$ e velocità di compressione $[vz_{fl}, vz_{fr}, vz_{rl}, vz_{rr}]$
  - 4 Ruote: velocità angolari di spin $[\omega_{fl}, \omega_{fr}, \omega_{rl}, \omega_{rr}]$
  - 8 Stati di rilassamento elastico della carcassa pneumatico: $[u_{x,fl}, u_{y,fl}, u_{x,fr}, u_{y,fr}, u_{x,rl}, u_{y,rl}, u_{x,rr}, u_{y,rr}]$
- **Caratteristiche Avanzate & Architettura Multibody Newton III**:
  - **Dinamica 2° Ordine Masse Non Sospese & Carico $F_z$ Emergente**:
    La deflessione a terra del pneumatico $\delta_{z,\text{tire},i} = \delta_{\text{tire},0,i} + z_{\text{road},i} - z_{w,i}$ è rigorosamente isolata dal moto della cassa, impedendo anelli parassiti di amplificazione energetica. Il carico normale a terra emerge da:
    $$F_{z,i} = \min\left(\max\left(K_{tz}\delta_{z,\text{tire},i} - C_{tz} v_{z,w,i}, 0.0\right), F_{z,\max}\right)$$
    La dinamica di ciascun mozzo ruota è governata dalla 2ª legge di Newton:
    $$m_{u,i} \ddot{z}_{w,i} = F_{z,\text{tire},i} - F_{\text{susp},i} - m_{u,i} g$$
  - **Cinematica Relativa Corner-Mozzo & Velocità di Corsa**:
    Punti attacco telaio:
    $$z_{\text{chassis},i} = z_s - \theta x_{c,i} + \phi y_{c,i}, \quad \dot{z}_{\text{chassis},i} = w_{\text{heave}} - \dot{\theta} x_{c,i} + \dot{\phi} y_{c,i}$$
    Corsa relativa (compressione/jounce):
    $$z_{\text{jounce},i} = z_{w,i} - z_{\text{chassis},i}, \quad v_{\text{jounce},i} = \dot{z}_{w,i} - \dot{z}_{\text{chassis},i}$$
  - **Accoppiamento Azione-Reazione Diretto (Newton III)**:
    Le forze di sospensione (molle non lineari con bumpstop progressivo, barre ARB anti-simmetriche e smorzatori TTX25 MkII) generano forze di compressione $F_{\text{susp},i}$ che spingono verso l'alto la cassa ($+F_{\text{susp},i}$) e verso il basso il mozzo ($-F_{\text{susp},i}$):
    $$F_{\text{susp},i} = F_{\text{susp},0,i} + K_{\text{eff},i} z_{\text{jounce},i} + F_{\text{damp},i}(v_{\text{jounce},i}) + F_{\text{arb},i}$$
    L'equilibrio dinamico della cassa è chiuso:
    $$m_s \dot{w}_{\text{heave}} = \sum_{i=1}^4 F_{\text{susp},i} - F_{\text{down}} - m_s g + m_s(u q - v p)$$
    $$M_{x,\text{susp}} = -\frac{t_f}{2}(F_{\text{susp},fr} - F_{\text{susp},fl}) - \frac{t_r}{2}(F_{\text{susp},rr} - F_{\text{susp},rl})$$
    $$M_{y,\text{susp}} = -a(F_{\text{susp},fl} + F_{\text{susp},fr}) + b(F_{\text{susp},rl} + F_{\text{susp},rr})$$
  - **Garanzia di Dissipazione Energetica**:
    La formulazione assicura identicamente $\dot{E}_{\text{mech}} \le P_{\text{in}}$ senza alcuna iniezione numerica di lavoro spurio ($\max\text{Re}(\lambda) = -0.0718$ in rettilineo a 15 m/s, dissipazione continua $\dot{E} \le -2050\text{ W}$ in coast-down ad assetto transitorio).
  - **Convenzione $z_i = 0$ all'equilibrio statico a terra**: Tutte le grandezze di assetto (altezza nominale, campanatura $\gamma_0$, convergenza $\delta_{toe,0}$) sono calibrate a vettura ferma a terra con pilota a bordo (1g statico), garantendo perfetta continuità all'avvio ($t=0$).
  - **Trasferimento di Carico Longitudinale Dinamico (Pitch Load Transfer)**: In accelerazione e frenata, la cassa subisce squat/dive rotazionale attorno all'asse di beccheggio. Il trasferimento di carico tra assale anteriore e posteriore è calcolato rigorosamente da rigidezza e smorzamento in beccheggio:
    $$dF_{z,\text{long}} = \frac{K_\theta(-\theta) + C_\theta(-\dot{\theta})}{L_{\text{sym}}}$$
    dove $K_\theta = 2 K_{z,f} a^2 + 2 K_{z,r} b^2$ e $C_\theta = 2 C_{z,f} a^2 + 2 C_{z,r} b^2$. In piena trazione (squat $\theta \approx -0.49^\circ$), il carico al retrotreno aumenta dinamicamente da $1475\text{ N}$ a oltre **$2003\text{ N}$** conservando rigorosamente la forza normale totale.
  - **Integrazione Antifeature Sospensioni (Anti-Dive, Anti-Squat, Anti-Lift)**: I polinomi cinematici di 3° grado $AD(z_f)$, $AS(z_r)$ e $AL(z_r)$ (derivati dai rilievi Adams Car) ripartiscono geometricamente il carico longitudinale. Una quota $AS \cdot \Delta F_{z,\text{long}}$ in accelerazione (o $AD/AL$ in frenata) è retta rigidamente dai triangoli della sospensione riducendo proporzionalmente il momento ribaltante sulle molle ($M_{\text{pitch}}$), mentre a terra i pneumatici ricevono il 100% del carico trasferito.
  - **Modello di Rilassamento Dissipativo della Patch di Contatto & Clamping Combined Slip**: La frequenza di rilassamento elastico della carcassa è governata dalla velocità planare totale del mozzo $V_{mag} = \sqrt{V_{xw}^2 + V_{yw}^2 + \epsilon_v^2}$, impedendo l'accumulo di deflessioni fittizie in derapata laterale. Le deflessioni sono saturate al limite di slittamento Coulombiano ($1.5 \sigma$) tramite funzione liscia $\tanh$, e i fattori di combined slip $G_{xa}, G_{yk}$ sono vincolati a valori non-negativi ($\ge 0$), garantendo dissipazione rigorosa dell'energia cinetica in testacoda con decadimento dell'imbardata a zero.
  - **Disaccoppiamento Inerzie Ruote ($I_{w,f} = 0.30\text{ kg}\cdot\text{m}^2$ vs $I_{w,r} = 1.20\text{ kg}\cdot\text{m}^2$)**: Per veicoli RWD, l'avantreno non motrice ha un'inerzia propria ridotta ($I_{w,f} = 0.30\text{ kg}\cdot\text{m}^2$), eliminando oltre $400\text{ N}$ di coppia frenante parassita da accelerazione angolare all'anteriore e liberando $+0.15g$ di trazione utile.
  - **Accoppiamento Pressione di Esercizio Pneumatici**: L'ODE e il solutore accoppiano la pressione nominale reale ($0.85\text{ bar} = 12.3\text{ psi}$ vs $18.1\text{ psi}$ da banco) alle equazioni di forza longitudinale MF6.1, espandendo l'impronta a terra e incrementando l'aderenza $\mu_x$ del $+46.3\%$ tramite il coefficiente `PPX3`.
  - **Controllo di Trazione Disinseribile (Switchable TCS) con Derating Laterale**: Integrato algoritmo closed-loop a modulazione progressiva con target di scorrimento calibrato a $\kappa_{\text{target}} = 0.095$ (punto di picco $\mu_x$ del compound R20) e derating con l'angolo volante $\kappa_{\text{target}} = \max(0.095 - 0.045 \cdot (|\delta| / 15^\circ), 0.04)$ per preservare fino al 70% di grip laterale $F_y$ e impedire il testacoda in accelerazione. Attivabile da tastiera (tasto `T`) o GUI, include indicatore dinamico HUD (`[TCS: ATTIVO]`, `[TCS: -XX%]`, `[TCS: OFF]`).
  - **Refactoring Barre Antirollio (Lumped Stiffness & Wheel Rate)**:
    Disaccoppiata la dipendenza da geometrie e materiali cilindrici. L'utente o il tool di dimensionamento definisce direttamente la rigidezza elastica a rollio $K_{\phi,\text{arb}}$ [N·m/rad] o il wheel rate differenziale $K_{w,\text{arb}} = K_{\phi,\text{arb}} / t^2$ [N/m]. Nel modello 14-DOF le forze verticali sui puntoni sono anti-simmetriche:
    $$F_{\text{arb},fl} = +K_{w,\text{arb}}\Delta z_{\text{susp}}, \quad F_{\text{arb},fr} = -K_{w,\text{arb}}\Delta z_{\text{susp}}$$
    generando una forza di heave netta nulla ($\sum F_{\text{arb}} = 0$) e il momento ripristinante esatto $M_{x,\text{arb}} = -K_{\phi,\text{arb}}(\phi - \phi_{\text{axle}})$.
  - **Parametrizzazione Fisica Molle Sospensione**:
    Esposizione in `VehicleParams` di lunghezza libera a riposo $L_0$ (`spring_free_length_f/r_mm`), precarico installato $\Delta L_{\text{pre}}$ (`spring_preload_f/r_mm`) e lunghezza a massima estensione/droop $L_{\text{inst}} = L_0 - \Delta L_{\text{pre}}$. La forza complessiva di molla include il precarico statico: $F_{\text{spring}} = (F_{z,0} - m_u g) + K_{\text{eff}} z_{\text{jounce}}$.
  - **Telecamere Onboard & Stabilizzazione Inerziale (Interactive Driver 14-DOF)**:
    Telecamere Onboard (Cockpit, Nose, Bumper) rigidamente ancorate al frame telaio senza ritardo EMA in coordinate mondo, prevenendo lo slittamento visivo della camera fuori dall'abitacolo durante forti accelerazioni e frenate. Per la camera Chase, la direzione e l'offset verticale compensano la pendenza longitudinale del profilo stradale (`grade_rd`), impedendo l'inclinazione visiva verso il manto d'asfalto in salita. Per tutti i preset di camera è attivo l'unwrapping circolare a cammino minimo per l'angolo di imbardata $\psi \in [-\pi, \pi]$ ($d\psi = (\psi - \psi_{\text{smooth}} + \pi) \pmod{2\pi} - \pi$), scongiurando qualsiasi spin a 360° durante inversioni di marcia o passaggi al limite di discontinuità angolare.
  - **Aggiornamento Dinamico Pista 3D & Cache Invalidation OpenGL**:
    Integrazione del contatore atomico di versione `_version` in `road_profile.py` e del metodo `invalidate_track_display_list()` in `interactive_driver_14dof.py`. Qualsiasi variazione dei parametri altimetrici, dossi sintetici o banking da GUI/Track Designer invalida istantaneamente il display list precompilato dell'asfalto, rigenerando la mesh tridimensionale. La quota del piano erboso è dinamicamente calcolata come $z_{\text{ground}} = \min(z_{\text{car}}, \min(z_{\text{cnt}})) - 0.25\text{ m}$, impedendo che il terreno verde sommerga salite e cordoli sopraelevati.
  - **Convergenza e Robustezza Speed Sweep 14-DOF (`mdo_global_optimizer.py`)**:
    Risolto il problema del collasso dell'accelerazione laterale massima a $0.0\text{g}$. Il solutore implementa:
    1. Risoluzione sicura dei percorsi `.tir` con fallback automatico al file nominale Hoosier se la stringa è vuota, punta a directory o file non accessibile;
    2. Inizializzazione cinematica ad assetto variabile per le variabili di trim: $\delta_{\text{kin},0} = \arctan(L/R)$, $v_{y,\text{kin},0} = -b/R \cdot V_x$, con altezze telaio $z_{f,0}, z_{r,0}$ pre-deflessionate in base alla downforce aerodinamica a velocità $V_x$;
    3. Validazione rigorosa della norma del residuo su entrambi i solutori (`hybr` e `lm`) tramite $\| \text{res} \| \le \text{res\_limit}$, rigettando false convergenze su gradienti piatti;
    4. Criterio di arresto controsterzo tollerante $\Delta\delta < -1.5^\circ$ per evitare terminazioni su micro-ondulazioni numeriche;
    5. Fallback fisico analitico aero-potenziato di sicurezza contro singolarità locali.
  - **Banking Stradale ISO 8855 & Quota Patch di Contatto 3D**:
    Convenzione ISO 8855 (+X avanti, +Y sinistra, +Z alto): inclinazione a destra $\phi_{\text{bank}} > 0$ genera componente gravitazionale laterale $a_{y,\text{road\_grav}} = -g \sin(\phi_{\text{bank}})$. La quota delle 4 impronte a terra tiene conto dell'offset laterale effettivo $d_i$:
    $$z_{\text{road},i} = z_{\text{centerline}}(s_i) + d_i \sin(\phi_{\text{bank}}(s_i))$$
  - **Sottosistema Tattile LT/RT (Rate-Limiting 50-60 Hz & Isteresi Bloccaggio)**:
    Disaccoppiamento della chiamata di vibrazione a 50–60 Hz ($\Delta t \ge 16\text{ ms}$) con soglia minima di sensibilità $\Delta\text{intensity} > 0.02$ per evitare la saturazione dell'endpoint USB HID a 250–500 Hz. Isteresi su slip ratio di bloccaggio ABS ($\kappa_{\text{on}} = 0.12$, $\kappa_{\text{off}} = 0.08$) con tempo minimo di hold di 80 ms per azzerare il chatter sui trigger fisici.
- Cinematica sospensioni non-lineare con recupero camber e toe in funzione del rollio e del travel.
- Altezze centri di rollio verticali $h_{rc,f}, h_{rc,r}$ e migrazione laterale $a_1$ accoppiata al rollio $\phi$.
- Ripartizione del trasferimento di carico laterale (LLTD) dinamica accoppiata con barre antirollio anteriori/posteriori ($K_{\phi,f}, K_{\phi,r}$).
- Mappa aerodinamica a 3 assi con spostamento del centro di pressione (CoP) e variazione di downforce/drag con l'altezza da terra.

---

## 3. Modello di Pneumatico Pacejka MF6.1 (`tire_model_mf61.py`)

I pneumatici sono modellati con la formulazione analitica e simbolica **Pacejka Magic Formula 6.1 (MF6.1)** in piena conformità al manuale ufficiale *Simcenter Tire MF-Tool Equation Manual (FITTYP = 61)*:

### 3.1 Dipendenza da Carico e Pressione di Gonfiaggio
- **Carico Nominale Scalato**: $F'_{z0} = \lambda_{Fz0} \cdot F_{z0}$ (dove $\lambda_{Fz0} = \text{LFZO}$).
- **Incremento di Carico Normale**: $df_z = \frac{F_z - F'_{z0}}{F'_{z0}}$.
- **Incremento di Pressione di Gonfiaggio**: $dp_i = \frac{P_i - P_{nom}}{P_{nom}}$.

### 3.2 Forza Laterale Pura ($F_{y0}$)
$$F_{y0} = D_y \sin\left[ C_y \arctan\left( B_y \alpha_y - E_y (B_y \alpha_y - \arctan(B_y \alpha_y)) \right) \right] + S_{Vy}$$
- $C_y = p_{Cy1} \cdot \lambda_{Cy}$ (Fattore di forma).
- $\mu_y = (p_{Dy1} + p_{Dy2} df_z)(1 + p_{py3} dp_i + p_{py4} dp_i^2)(1 - p_{Dy3} \gamma^2) \lambda_{\mu y}$ (Coefficiente d'attrito).
- $D_y = \mu_y \cdot F_z$ (Fattore di picco).
- $K_{y\alpha} = p_{Ky1} F'_{z0} (1 + p_{py1} dp_i) \sin\left[ p_{Ky4} \arctan\left( \frac{F_z}{(p_{Ky2} + p_{Ky5} \gamma^2) F'_{z0} (1 + p_{py2} dp_i)} \right) \right] (1 - p_{Ky3} |\gamma|) \lambda_{Ky\alpha}$ (Rigidezza di deriva a $\alpha=0$).
- $B_y = \frac{K_{y\alpha}}{C_y D_y}$ (Fattore di rigidezza).
- $K_{y\gamma 0} = (p_{Ky6} + p_{Ky7} df_z) F_z \lambda_{Ky\gamma} (1 + p_{py5} dp_i)$ (Rigidezza di campanatura).
- $S_{Vy} = F_z (p_{Vy1} + p_{Vy2} df_z) \lambda_{Vy} \lambda_{\mu y} + F_z (p_{Vy3} + p_{Vy4} df_z) \gamma \lambda_{Ky\gamma} \lambda_{\mu y}$ (Shift verticale).
- $S_{Hy} = (p_{Hy1} + p_{Hy2} df_z) \lambda_{Hy} + \frac{K_{y\gamma 0} \gamma - S_{Vy\gamma}}{K_{y\alpha}}$ (Shift orizzontale con Camber Thrust).
- $\alpha_y = \alpha + S_{Hy}$ (Angolo di deriva effettivo traslato).
- $E_y = (p_{Ey1} + p_{Ey2} df_z)[1 + p_{Ey5} \gamma^2 - (p_{Ey3} + p_{Ey4} \gamma) \operatorname{sgn}(\alpha_y)] \lambda_{Ey} \le 1$ (Fattore di curvatura).

### 3.3 Forza Longitudinale Pura ($F_{x0}$)
$$F_{x0} = D_x \sin\left[ C_x \arctan\left( B_x \kappa_x - E_x (B_x \kappa_x - \arctan(B_x \kappa_x)) \right) \right] + S_{Vx}$$
- $C_x = p_{Cx1} \cdot \lambda_{Cx}$, $D_x = \mu_x \cdot F_z$.
- $\mu_x = (p_{Dx1} + p_{Dx2} df_z)(1 + p_{px3} dp_i + p_{px4} dp_i^2)(1 - p_{Dx3} \gamma^2) \lambda_{\mu x}$.
- $K_{x\kappa} = F_z (p_{Kx1} + p_{Kx2} df_z) \exp(p_{Kx3} df_z)(1 + p_{px1} dp_i + p_{px2} dp_i^2) \lambda_{Kx\kappa}$.
- $B_x = \frac{K_{x\kappa}}{C_x D_x}$, $S_{Hx} = (p_{Hx1} + p_{Hx2} df_z)\lambda_{Hx}$, $S_{Vx} = F_z (p_{Vx1} + p_{Vx2} df_z)\lambda_{Vx}\lambda_{\mu x}$.
- $\kappa_x = \kappa + S_{Hx}$, $E_x = (p_{Ex1} + p_{Ex2} df_z + p_{Ex3} df_z^2)(1 - p_{Ex4}\operatorname{sgn}(\kappa_x))\lambda_{Ex} \le 1$.

### 3.4 Slittamento Combinato (Combined Slip $F_x, F_y$)
- $F_x = F_{x0} \cdot G_{x\alpha}$, con fattore di ponderazione $G_{x\alpha}(\kappa, \alpha, \gamma, F_z)$.
- $F_y = F_{y0} \cdot G_{y\kappa} + S_{Vy\kappa}$, con fattore $G_{y\kappa}(\alpha, \kappa, \gamma, F_z)$ e shift indotto da $\kappa$ $S_{Vy\kappa}$.

### 3.5 Momento di Autoallineamento ($M_z$), Ribaltamento ($M_x$) e Resistenza al Rotolamento ($M_y$)
- **Momento di Autoallineamento $M_z$**:
  $$M_z = -t(\alpha_{t,eq}) \cdot F'_{y0} + M_{zr}(\alpha_{r,eq}) + s \cdot F_x$$
  dove $t$ è il braccio pneumatico (*pneumatic trail*), $M_{zr}$ è il momento residuo di camber e spin, e $s \cdot F_x$ è il momento asimmetrico della forza longitudinale.
- **Momento di Ribaltamento $M_x$ (Overturning Couple)**: Calcolato secondo Sez. 1.4 MF 6.1 tenendo conto di $F_y, \gamma, F_z, dp_i$.
- **Momento di Resistenza al Rotolamento $M_y$**: Calcolato secondo Sez. 1.5 MF 6.1 in funzione di $V_x, F_x, \gamma, F_z, dp_i$.

### 3.6 Convenzione Speculare del Pneumatico & Gestione Parametro `TYRESIDE`
Nel sistema di coordinate veicolo ISO ($+X$ in avanti, $+Y$ verso sinistra, $+Z$ verso l'alto):
- **Spinta di Campanatura (Camber Thrust)**: Una campanatura statica negativa ($c < 0$, sommità della ruota inclinata verso la mezzeria del veicolo) genera spinte laterali opposte sui due lati:
  - Ruota sinistra (FL/RL): si inclina verso destra $\implies$ il camber thrust spinge verso destra ($F_{y,\text{left}} < 0$).
  - Ruota destra (FR/RR): si inclina verso sinistra $\implies$ il camber thrust spinge verso sinistra ($F_{y,\text{right}} > 0$).
- **Parametro `TYRESIDE` nel file `.tir`**:
  I file Pacejka `.tir` (es. Calspan / FSAE TTC) sono fittati con riferimento al lato specifico del banco prova indicato dalla chiave `TYRESIDE = 'RIGHT'` oppure `'LEFT'`:
  - Se `TYRESIDE == 'RIGHT'` (es. Hoosier 43075): i coefficienti descrivono direttamente la ruota destra. Per la ruota sinistra, la campanatura effettiva deve essere invertita ($\gamma_{\text{left}} = -c$), così come i coefficienti di asimmetria e conicità/ply-steer ($\lambda_{Vy,\text{left}} = -1.0$, $\lambda_{Hy,\text{left}} = -1.0$).
  - Se `TYRESIDE == 'LEFT'` (es. SC24): la ruota sinistra assume direttamente $\gamma_{\text{left}} = c$, mentre la ruota destra viene specchiata ($\gamma_{\text{right}} = -c$, $\lambda_{Vy,\text{right}} = -1.0$, $\lambda_{Hy,\text{right}} = -1.0$).
- Questa logica è integrata nativamente in `TireParams.get_tyreside()`, in tutti i solutori (`Vehicle14DOF`, `Vehicle7DOF`, `Vehicle3DOF`, `mdo_dynamic_events.py`, `mdo_target_optimization.py`) e nella telemetria real-time (`interactive_driver_14dof.py`), azzerando qualsiasi forza laterale netta o imbardata parassita in rettilineo.

### 3.7 Specchiatura nello Slittamento Combinato (Combined Slip Lateral & Longitudinal Mirroring)
Nelle manovre di forte trazione o frenata in rettilineo ($\kappa \ne 0$), il modello di slittamento combinato Pacejka MF6.1 introduce traslazioni orizzontali e verticali:
- **Shift Orizzontale in Deriva ($B_{yk}$)**: $\alpha_{y,\text{eff}} = \alpha - RBY_3 \cdot \lambda_{Hy}$.
- **Shift Orizzontale Longitudinale ($S_{Hxa}$)**: $S_{Hxa} = RHX_1 \cdot \lambda_{Hy}$.
- **Forza Laterale Indotta da $\kappa$ ($S_{Vy\kappa}$)**:
  $$D_{Vyk} = \mu_y F_z \left( (RVY_1 + RVY_2 \cdot df_z)\lambda_{Vy} + RVY_3 \gamma \right) \cos\left[\arctan(RVY_4 \alpha)\right]$$
  $$S_{Vy\kappa} = D_{Vyk} \sin\left[r_{Vy5} \arctan(r_{Vy6} \kappa)\right] \lambda_{Vy\kappa}$$
Poiché i coefficienti del banco di prova (es. $RVY_1 = 0.013365, RVY_3 = 1.1393, RHX_1 = -0.012984$) sono calibrati sul singolo pneumatico asimmetrico, omettere la specchiatura con $\lambda_{Vy}$ e $\lambda_{Hy}$ produce uno shift concorde ($S_{Vy\kappa} > 0$ su entrambe le ruote posteriori durante l'accelerazione), rompendo la simmetria dinamica.  
Con l'introduzione dei moltiplicatori $\lambda_{Vy}, \lambda_{Hy}$ in `mf61_combined_lateral_force`, `mf61_combined_longitudinal_force` e `mf61_aligning_torque`:
$$F_{y,rl}(\kappa > 0) = -F_{y,rr}(\kappa > 0), \quad \Sigma F_{y,\text{axle}} = 0.00\text{ N}, \quad \Sigma M_{z,\text{axle}} = 0.00\text{ Nm}$$

### 3.8 Convenzione Angoli di Toe (Toe-Out Anteriore vs Toe-In Posteriore)
Per garantire coerenza fisica tra i parametri di setup (`params_vehicle.py`) e le equazioni cinematiche:
- **Avantreno (`toe_f_static > 0`)**: Convenzione standard **Toe-Out** (divergenza) per migliorare la prontezza di inserimento in curva:
  $$\alpha_{fl} = \alpha_{\text{geom}, fl} + \delta_{\text{toe}, f}, \quad \alpha_{fr} = \alpha_{\text{geom}, fr} - \delta_{\text{toe}, f}$$
- **Retrotreno (`toe_r_static > 0`)**: Convenzione standard **Toe-In** (convergenza) per massimizzare la stabilità direzionale:
  $$\alpha_{rl} = \alpha_{\text{geom}, rl} - \delta_{\text{toe}, r}, \quad \alpha_{rr} = \alpha_{\text{geom}, rr} + \delta_{\text{toe}, r}$$
Grazie a questo segno, sia la campanatura statica negativa ($\gamma < 0$) che la convergenza posteriore positiva ($\delta_{\text{toe}, r} > 0$) generano forze convergenti verso l'interno del veicolo ($F_{y,\text{left}} < 0$, $F_{y,\text{right}} > 0$), che si elidono esattamente a vicenda sull'asse.

---

## 4. Ottimizzatore Globale MDO & Derivate di Stabilità (`mdo_global_optimizer.py`)

L'ottimizzatore risolve il problema di sintesi ottimale dell'assetto del veicolo (3-DOF, 7-DOF e 14-DOF) massimizzando le performance e garantendo il controllo del comportamento di manovrabilità:

### 4.1 Algoritmo Numerico & Formulazione degli Obiettivi
- **Solutore**: SLSQP (Sequential Least Squares Programming) con ricerca **Multi-Start** su una griglia di punti iniziali diversificati per evitare minimi locali non ottimali.
- **Modalità Obiettivo Supportate**:
  1. `max_ay`: Massimizzazione dell'accelerazione laterale limite pura:
     $$J(X) = -a_y(X) + \text{penalty}_{\text{geom}}(X) + \text{penalty}_{us}(X)$$
  2. `target_neutral`: **Assetto Neutro Bilanciato** (Target $K_{us}$):
     $$J(X) = -a_y(X) + w_{us} \cdot 1.0 (K_{us}(X) - K_{us,\text{target}})^2 + \text{penalty}_{\text{geom}}(X)$$
     Questa formulazione consente di ottenere la massima aderenza laterale possibile vincolando o guidando il gradiente di sottosterzo verso il valore ideale (es. $K_{us} \approx 0.0^\circ/\text{g}$ o debolmente positivo $0.05^\circ/\text{g}$), evitando assetti violentemente sovrasterzanti che risulterebbero inguidabili per il pilota.
  3. `event_autox` / `event_skidpad` / `event_accel`: Ottimizzazione basata sui tempi stimati nei singoli eventi dinamici FSAE.

### 4.2 Calcolo Rigoroso del Gradiente di Sottosterzo ($K_{us}$) ed Eliminazione del Camber Thrust
Nelle formulazioni analitiche di base, la rigidezza d'angolo equivalente $C_\alpha = \frac{\partial F_y}{\partial \alpha}$ deve riflettere la sola rigidezza di deriva dello pneumatico.
Nei modelli Pacejka MF6.1 completi, la presenza di una campanatura statica ($\gamma_0 = -1.5^\circ$ all'anteriore, $-1.0^\circ$ al posteriore) genera una forza di spinta da campanatura (*camber thrust*) anche ad angolo di deriva nullo ($\alpha = 0$):
$$F_y(\alpha=0, \gamma_0) \approx -490\text{ N}$$
Inoltre, le asimmetrie costruttive o di test del battistrada introducono piccoli offset di conicità/ply-steer ($S_{Vy0} \approx 10\text{ N}$). Se si calcola la derivata numerica con incrementi microscopici ($d_\alpha = 10^{-4}\text{ rad}$), il rapporto incrementale viene dominato dal gradiente locale di camber thrust e offset, portando a stime falsamente iper-sovrasterzanti ($K_{us} \approx -0.71^\circ/\text{g}$).
Per garantire un calcolo fisicamente puro e non corrotto:
1. Viene imposto l'azzeramento della campanatura $\gamma = 0$ durante la valutazione della derivata $\partial F_y / \partial \alpha$.
2. L'incremento di angolo di deriva per la corda di rigidezza è calibrato a $d_\alpha = 1.0^\circ$, integrando il comportamento reale dello pneumatico nel range lineare:
   $$C_{\alpha,i} = \frac{F_{y,i}(d_\alpha, \gamma=0) - F_{y,i}(0, \gamma=0)}{d_\alpha}$$
3. Con questa formulazione, il veicolo nominale FSAE presenta un gradiente pulito e coerente $K_{us} \approx -0.11^\circ/\text{g}$.

### 4.3 Accoppiamento Rigidezze di Rollio 14-DOF ($K_{\phi,f}, K_{\phi,r}$)
Nei modelli a 14 gradi di libertà, i parametri di rigidezza angolare delle barre antirollio e delle molle di sospensione $K_{\phi,f}, K_{\phi,r}$ determinano la ripartizione elastica del momento di rollio ($LLTD_{\text{elastic}}$):
$$LLTD_{\text{elastic}} = \frac{K_{\phi,f}}{K_{\phi,f} + K_{\phi,r}}$$
In curva a regime ($a_y = 1\text{g}$), il trasferimento di carico laterale sulle singole ruote viene modulato direttamente da $LLTD_{\text{elastic}}$:
$$\Delta F_{z,f} = \frac{m \cdot a_y \cdot h_{cg} \cdot LLTD_{\text{elastic}}}{t_f}, \quad \Delta F_{z,r} = \frac{m \cdot a_y \cdot h_{cg} \cdot (1 - LLTD_{\text{elastic}})}{t_r}$$
I carichi verticali effettivi interni ed esterni determinano la saturazione non-lineare e le rigidezze $C_{\alpha,f}, C_{\alpha,r}$. L'irrigidimento dell'avantreno aumenta la saturazione anteriore e accresce il sottosterzo ($K_{us} \uparrow$), mentre l'irrigidimento del retrotreno promuove il sovrasterzo ($K_{us} \downarrow$).

### 4.4 Condizione di Arresto al Picco Fisico di Sterzata & Frontiera di Stabilità
Nell'analisi di aderenza limite a velocità costante ($v_x = \text{cost}$):
- In un veicolo sovrasterzante o con saturazione progressiva del retrotreno, l'angolo di sterzo $\delta$ aumenta con $a_y$ fino a raggiungere un picco di manovra stazionaria $\left(\frac{\partial a_y}{\partial \delta}\right)_{v_x} = 0$.
- Oltre tale picco, le equazioni di equilibrio ammettono soluzioni puramente matematiche in cui l'accelerazione laterale cresce leggermente al costo di una controsterzata ($\delta$ decrescente). Queste soluzioni appartengono al ramo instabile e contronaturale del veicolo.
- In `_evaluate_limit_ay_at_speed`, la ricerca dell'accelerazione laterale massima arresta la progressione non appena viene rilevata l'inversione dello sterzo:
  $$\delta_{k} < \delta_{k-1} - 10^{-4}\text{ rad} \quad (a_y > 0.5\text{g})$$
  impedendo all'ottimizzatore di premiare assetti fittiziamente stabili basati su equilibri post-saturazione.
- **Frontiera di Stabilità nell'Handling Diagram**: Nel diagramma di manovrabilità di Pacejka $(\delta - L/R \text{ vs } a_y)$, la curva viene tracciata mantenendo il metodo di continuazione della lunghezza d'arco (arc-length) per visualizzare l'intero ramo discendente di perdita di controllo. La sovrapposizione opzionale della **Frontiera di Stabilità** congiunge i picchi $(\partial a_y / \partial\delta = 0)$ calcolati a tutte le velocità di sweep, identificando istantaneamente il confine oltre il quale il veicolo perde la stabilità a catena cinematica aperta.

### 4.5 Calibrazione Graduale delle Penalità
La penalizzazione per violazione del vincolo di sottosterzo $K_{us} < K_{us,\min}$ è modellata con progressione quadratica scalata:
$$\text{penalty}_{us} = 100 \cdot (\min(0, K_{us} - K_{us,\min}))^2$$
Questa calibrazione previene l'effetto di "schiacciamento sui bordi" (boundary pinning) in cui le variabili di ripartizione masse e carico venivano forzate ai limiti geometrici estremi ($wd_f = 65\%, LLTD = 65\%$).

### 4.6 Derivate di Stabilità Linearizzate & Analisi Modale
- $Y_\beta = -(C_{\alpha,f} + C_{\alpha,r})$ $[\text{N/rad}]$
- $Y_r = \frac{-(a C_{\alpha,f} - b C_{\alpha,r})}{v_x}$ $[\text{N}\cdot\text{s/rad}]$
- $N_\beta = -(a C_{\alpha,f} - b C_{\alpha,r})$ $[\text{Nm/rad}]$ *(Stabilità direzionale statica / Effetto banderuola)*
- $N_r = \frac{-(a^2 C_{\alpha,f} + b^2 C_{\alpha,r})}{v_x}$ $[\text{Nm}\cdot\text{s/rad}]$ *(Smorzamento naturale d'imbardata)*
- $N_\delta = a C_{\alpha,f}$ $[\text{Nm/rad}]$ *(Autorità dello sterzo)*
- Frequenza naturale $\omega_n = \sqrt{\det(A)}$ e smorzamento adimensionale $\zeta = -\frac{\operatorname{tr}(A)}{2\omega_n}$.
- Dynamic Index: $DI = \frac{I_z}{m \cdot a \cdot b}$.

---

## 5. Motore di Sensitività Transitoria Parametrica & Sweep 1D/2D (`src/utils/transient_sensitivity.py`)

Modulo dedicato all'analisi di sensitività parametrica deterministica e transitoria non-lineare per prototipi da corsa (conforme a standard ISO 7401, ISO 3888-2, SAE J266 e OptimumG Tech Tips #28 & #30).

### 5.1 Il Catalogo degli Scenari Parametrici (`SCENARIO_CATALOG`)
Lo sweep 1D/2D consente di isolare l'effetto di singoli parametri geometrici ed elastici di telaio e sospensioni mantenendo la coerenza fisica globale (es. ricalibrazione automatica dei polinomi di roll center, sincronizzazione tra rigidezze d'asse e LLTD elastico):

1. **`rc_height_f` (Front Roll Center Height $q_1$ $[0.02 \div 0.14\text{ m}]$)**:
   Altezza baricentrica del centro di rollio anteriore. Aumentando $q_1$ cresce il trasferimento di carico laterale per via geometrica pura (*direct force / non-suspended load transfer*), riducendo il braccio ribaltante $h_{\text{arm}} = h_{cg} - h_{ra}$ e contenendo l'angolo di rollio $\phi_{\max}$. Modifica direttamente il coefficiente costante $a_0$ del polinomio verticale Adams Car.
2. **`rc_height_r` (Rear Roll Center Height $q_2$ $[0.02 \div 0.16\text{ m}]$)**:
   Altezza del centro di rollio posteriore. Un $q_2$ più alto incrementa il trasferimento geometrico al retrotreno, aumentando la saturazione degli pneumatici posteriori in ingresso curva e incrementando la stabilità (tendenza sottosterzante).
3. **`rc_lat_gain_f` (Front RC Lateral Migration Gain $a_1$ $[0.0 \div 2.0\text{ m/m}]$)**:
   Guadagno lineare di migrazione trasversale del centro di rollio anteriore in funzione della differenza di travel $(z_{fl} - z_{fr})$. Valori elevati spostano rapidamente il polo verso la ruota esterna, generando un braccio asimmetrico e alterando il dynamic LLTD.
4. **`rc_lat_gain_r` (Rear RC Lateral Migration Gain $a_1$ $[0.0 \div 1.8\text{ m/m}]$)**:
   Guadagno lineare di migrazione trasversale del centro di rollio posteriore, con impatto prioritario sull'asimmetria di carico $F_z$ tra interno ed esterno curva al retrotreno.
5. **`arb_distribution` (Ripartizione Elastica Antirollio $\frac{K_{\phi,f}}{K_{\phi,\text{tot}}}$ $[0.30 \div 0.75]$)**:
   Ripartizione della rigidezza a rollio attribuita all'avantreno mantenendo rigorosamente costante la rigidezza totale $K_{\phi,\text{tot}} = K_{\phi,f} + K_{\phi,r}$.
   - $> 0.50$: Avantreno più rigido $\implies \text{LLTD}_f$ elevato $\implies$ sottosterzo di transitorio e stazionario.
   - $< 0.50$: Retrotreno più rigido $\implies$ sovrasterzo e maggiore agilità.
6. **`K_phi_f` (Rigidezza di Rollio Anteriore $K_{\phi,f}$ $[5\,000 \div 45\,000\text{ Nm/rad}]$)**:
   Rigidezza a rollio complessiva anteriore (molle di corner + barra ARB). Sincronizza automaticamente il parametro `vp.lltd`.
7. **`K_phi_r` (Rigidezza di Rollio Posteriore $K_{\phi,r}$ $[5\,000 \div 45\,000\text{ Nm/rad}]$)**:
   Rigidezza a rollio complessiva posteriore.
8. **`roll_stiffness_tot` (Rigidezza di Rollio Totale $K_{\phi,\text{tot}}$ $[4\,000 \div 22\,000\text{ Nm/rad}]$)**:
   Rigidezza complessiva a rapporto costante $\frac{K_{\phi,f}}{K_{\phi,r}}$. Riduce l'angolo di rollio globale preservando il bilanciamento stazionario nominale.
9. **`h_cg` (Altezza Baricentro da Terra $h_{cg}$ $[0.18 \div 0.40\text{ m}]$)**:
   Altezza del centro di massa. Parametro chiave di layout che moltiplica direttamente la forza d'inerzia laterale $m a_y h_{cg}$, determinando il momento ribaltante complessivo.
10. **`damping_phi` (Smorzamento Totale di Rollio $C_{\phi,\text{tot}}$ $[200 \div 3\,000\text{ Nms/rad}]$)**:
    Smorzamento rotazionale totale a rapporto costante assale anteriore/posteriore, calibrato per ottimizzare il fattore di smorzamento $\zeta_\phi$.

---

### 5.2 Estrazione dei 21 KPI Dinamici & Formule Matematiche
Per ogni punto dello sweep viene simulata la manovra (Step Steer a rampa o Double Lane Change ISO 3888-2) tramite il modello 14-DOF (o 7-DOF), estraendo 21 grandezze scalari ingegneristiche:

#### Gruppo A: Risposta d'Imbardata & Dinamica Laterale (Yaw Response)
1. **Steady-State Yaw Rate ($r_{ss}$ [deg/s])**:
   Velocità angolare d'imbardata a regime stazionario, calcolata come media negli ultimi 0.5 s della simulazione:
   $$r_{ss} = \frac{1}{\Delta T}\int_{t_{\text{end}}-0.5}^{t_{\text{end}}} r(t)\,dt$$
2. **Peak Yaw Rate ($r_{\text{peak}}$ [deg/s])**:
   Valore massimo assoluto raggiunto dalla velocità d'imbardata durante il transitorio:
   $$r_{\text{peak}} = \max_{t \ge t_{\text{step}}} |r(t)|$$
3. **Overshoot Percentage ($OS\%$ [%])**:
   Sovraelongazione percentuale dell'imbardata rispetto al valore stazionario:
   $$OS\% = \max\left(0, \frac{|r_{\text{peak}}| - |r_{ss}|}{|r_{ss}|} \times 100\right)$$
4. **Rise Time 90% ($t_{90}$ [s])**:
   Tempo necessario dall'istante di inizio sterzata $t_{\text{step}}$ per raggiungere il 90% del valore di regime:
   $$t_{90} = \min\left\{t > t_{\text{step}} \;\Big|\; |r(t)| \ge 0.90 |r_{ss}|\right\} - t_{\text{step}}$$
5. **Yaw Delay Time ($\Delta t_{\text{yaw}}$ [s])**:
   Ritardo temporale tra il 50% dell'input di sterzo e il 50% della risposta di regime in imbardata:
   $$\Delta t_{\text{yaw}} = t\left(|r| = 0.50 |r_{ss}|\right) - \left(t_{\text{step}} + \frac{t_{\text{ramp}}}{2}\right)$$
6. **Lateral Acceleration Delay Time ($\Delta t_{ay}$ [s])**:
   Ritardo temporale tra il 50% dell'input di sterzo e il 50% della risposta in accelerazione laterale:
   $$\Delta t_{ay} = t\left(|a_y| = 0.50 |a_{y,ss}|\right) - \left(t_{\text{step}} + \frac{t_{\text{ramp}}}{2}\right)$$
7. **Time-Beta Factor (TB-Factor [deg·s])**:
   Metrica normalizzata di stabilità transitoria (norma ISO 7401 / formulazione Crolla):
   $$\text{TB} = |\beta_{ss}| \cdot (t_{\text{peak}} - t_{\text{step}})$$
   Valori inferiori a $0.15\text{ deg}\cdot\text{s}$ denotano veicoli molto controllabili e stabili nei transitori rapidi.
8. **Peak Yaw Acceleration ($\dot{r}_{\text{peak}}$ [deg/s²])**:
   Accelerazione angolare massima d'imbardata, indice di prontezza e reattività iniziale del veicolo:
   $$\dot{r}_{\text{peak}} = \max_t |\dot{r}(t)|$$
9. **Settling Time ($t_{\text{settle}}$ [s])**:
   Tempo di assestamento entro la banda di tolleranza $\pm 2\%$ attorno a $r_{ss}$:
   $$t_{\text{settle}} = \min\left\{t \;\Big|\; \forall \tau \ge t, \; \frac{|r(\tau) - r_{ss}|}{|r_{ss}|} \le 0.02\right\} - t_{\text{step}}$$
10. **Yaw Steering Sensitivity Gain ($G_r$ [(°/s)/°])**:
    Guadagno di sterzo stazionario in imbardata:
    $$G_r = \frac{|r_{ss}|}{|\delta_{\text{steer}}|}$$
11. **Lateral Acceleration Steering Sensitivity ($G_{ay}$ [g/°])**:
    Guadagno di sterzo stazionario in accelerazione laterale:
    $$G_{ay} = \frac{|a_{y,ss}| / g}{|\delta_{\text{steer}}|}$$

#### Gruppo B: Rollio & Assetto Cassa
12. **Roll Gradient ($\text{RollGrad}$ [deg/g])**:
    Gradiente di rollio stazionario in funzione dell'accelerazione laterale:
    $$\frac{\partial \phi}{\partial a_y} = \frac{|\phi_{ss}|}{|a_{y,ss}| / g}$$
13. **Max Roll Angle ($\phi_{\max}$ [deg])**:
    Angolo di rollio massimo di picco durante il transitorio:
    $$\phi_{\max} = \max_t |\phi(t)|$$
14. **Peak Lateral Acceleration ($a_{y,\text{peak}}$ [g])**:
    Accelerazione laterale massima registrata dal baricentro:
    $$a_{y,\text{peak}} = \frac{\max_t |a_y(t)|}{g}$$
15. **Peak Vehicle Side-Slip Angle ($\beta_{\text{peak}}$ [deg])**:
    Angolo di assetto (slittamento laterale del baricentro $\beta = \arctan(v_y / v_x)$) massimo in valore assoluto:
    $$\beta_{\text{peak}} = \max_t |\beta(t)|$$

#### Gruppo C: Ripartizione Carichi, Attrito & Efficienza di Grip
16. **Dynamic LLTD at Peak Ay ($\text{LLTD}_{\text{peak}}$ [-])**:
    Ripartizione effettiva del trasferimento di carico laterale all'avantreno all'istante di picco di $a_y$:
    $$\text{LLTD}_{\text{peak}} = \frac{\Delta F_{z,f}}{\Delta F_{z,f} + \Delta F_{z,r}}\Bigg|_{t(a_{y,\text{peak}})} = \frac{|F_{z,fl} - F_{z,fr}|}{|F_{z,fl} - F_{z,fr}| + |F_{z,rl} - F_{z,rr}|}$$
17. **Steady-State Dynamic LLTD ($\text{LLTD}_{ss}$ [-])**:
    Ripartizione effettiva del trasferimento di carico laterale in regime stazionario stabilizzato.
18. **L/R Vertical Load Asymmetry ($\Delta F_{z,\text{asym}}\%$ [%])**:
    Asimmetria percentuale di trasferimento di carico laterale rispetto al valore teorico simmetrico, indotta dalla migrazione orizzontale dei centri di rollio:
    $$\Delta F_{z,\text{asym}}\% = \frac{|\Delta F_{z,\text{left}} - \Delta F_{z,\text{right}}|}{F_{z,\text{axle}}} \times 100$$
19. **Max Front Tire Utilization ($\text{Util}_{F,\max}$ [%])**:
    Grado massimo di saturazione d'aderenza del pneumatico anteriore più caricato rispetto all'ellisse d'attrito:
    $$\text{Util}_{F,\max} = \max_t \left(\frac{\sqrt{F_{x,f}^2 + F_{y,f}^2}}{\mu_y(F_{z,f}) \cdot F_{z,f}}\right) \times 100$$
20. **Max Rear Tire Utilization ($\text{Util}_{R,\max}$ [%])**:
    Grado massimo di saturazione d'aderenza del pneumatico posteriore più caricato.
21. **Effective Grip Efficiency ($\eta_\mu$ [%])**:
    Rapporto di efficienza dell'aderenza globale espresso come percentuale dell'accelerazione laterale di picco rispetto al coefficiente di attrito puro nominale $\text{LMUY}$:
    $$\eta_\mu = \frac{a_{y,\text{peak}}}{\text{LMUY}} \times 100\%$$

---

### 5.3 Modalità di Esecuzione & Gradienti di Sensibilità
- **Sweep 1D (`run_sensitivity_sweep_1d`)**:
  Esegue la sweep su una lista di $N$ punti discreti (default $N=11$). Produce una matrice dati `kpi_table` (DataFrame Pandas con $N$ righe e 21 colonne) e le telemetrie complete $t \mapsto [r, a_y, \phi, \text{LLTD}]$. Per ciascun KPI calcola il gradiente locale di sensibilità con differenze finite centrate:
  $$\frac{\partial \text{KPI}}{\partial P_i} \approx \frac{\text{KPI}(P_{i+1}) - \text{KPI}(P_{i-1})}{P_{i+1} - P_{i-1}}$$
- **Griglia 2D Bivariata (`run_sensitivity_sweep_2d`)**:
  Esegue l'esplorazione incrociata $N_X \times N_Y$ su due parametri indipendenti (es. $q_1 \times \text{LLTD}$), restituendo una mappa tridimensionale plottata tramite Heatmap con interpolazione bilineare e curve di livello iso-risposta.

---

## 6. Mappe del DNA del Veicolo & Definizione di Sottosterzo (`vehicle_dna_maps.py`)

Implementa la formulazione analitica rigorosa per le mappe stazionarie, gradienti di curvatura e stabilità direzionale (standard SAE J670e / ISO 4138):

### 6.1 Gradienti delle Mappe Stazionarie & Gradiente di Sottosterzo
Dall'equazione dello sterzo in regime stazionario:
$$\delta_v = L \rho_p + K a_y \iff \rho_p(\delta_v, a_y) = \frac{\delta_v}{L} - \frac{K}{L} a_y$$
I gradienti delle mappe stazionarie sono:
$$\operatorname{grad} \rho_p = \left( \frac{\partial \rho_p}{\partial a_y}, \frac{\partial \rho_p}{\partial \delta_v} \right) = (\rho_y, \rho_\delta) = \left( -\frac{K}{L}, \frac{1}{L} \right)$$
$$\operatorname{grad} \beta_p = \left( \frac{\partial \beta_p}{\partial a_y}, \frac{\partial \beta_p}{\partial \delta_v} \right) = (\beta_y, \beta_\delta) = \left( D_\beta, \frac{b}{L} \right)$$
Pertanto, il **gradiente intrinseco di sottosterzo** è rigorosamente:
$$\mathbf{K = -L \cdot \rho_y} \quad \left[\frac{\text{rad}}{\text{m/s}^2}\right] \text{ o } \left[\frac{^\circ}{\text{g}}\right]$$
- $\rho_y < 0 \iff K > 0$: Veicolo **Sottosterzante**.
- $\rho_y = 0 \iff K = 0$: Veicolo **Neutro**.
- $\rho_y > 0 \iff K < 0$: Veicolo **Sovrasterzante**.

La pendenza della curva di manovra a velocità costante $u = \text{cost}$ è:
$$\left(\frac{\partial \delta_v}{\partial a_y}\right)_{u=\text{cost}} = \frac{L}{u^2} + K = -\frac{\rho_y - \frac{1}{u^2}}{\rho_\delta}$$

### 6.2 Le 6 Mappe del DNA ($n_1 \dots n_6$) nel Piano $(\delta_v, a_y)$ (Sect. 7.11.4, Eq. 7.169 - 7.175)
Con $D = \beta_y \rho_\delta - \beta_\delta \rho_y < 0$:
1. $n_1(\delta_v, a_y) = 2\zeta\omega_n = -\operatorname{tr}(A) = \frac{1}{u_a D}(\hat{N}_\delta \beta_y - \hat{Y}_\delta \rho_y - \rho_\delta)$ *(Smorzamento Equivalente)*
2. $n_2(\delta_v, a_y) = \omega_n^2 = \det(A) = \frac{\hat{N}_\delta}{D}(\rho_y - \frac{1}{u_a^2})$ *(Rigidezza Dinamica Equivalente)*
3. $n_3(\delta_v, a_y) = -\frac{\hat{N}_\delta}{u_a^2}(\frac{\beta_\delta}{D} + u_a^2)$ *(Guadagno statico di assetto $\beta$ per $F_\beta$)*
4. $n_4(\delta_v, a_y) = \frac{\hat{Y}_\delta}{u_a}$ *(Termine impulsivo per $F_\beta$)*
5. $n_5(\delta_v, a_y) = -\frac{\hat{N}_\delta}{u_a^2}(\frac{\rho_\delta}{D})$ *(Guadagno statico di curvatura $\rho$ per $F_\rho$)*
6. $n_6(\delta_v, a_y) = \frac{\hat{N}_\delta}{u_a}$ *(Accelerazione angolare iniziale per $F_\rho$)*

### 6.3 Condizione di Stabilità Dinamica & Velocità Critica (Sect. 7.11.4 & 7.13, Eq. 7.172)
$$\det(A) = n_2 > 0 \iff \rho_y < \frac{1}{u_a^2} \iff -\frac{K}{L} < \frac{1}{u_a^2} \iff 1 + \frac{K u_a^2}{L} > 0$$
Nei veicoli sovrasterzanti ($K < 0 \iff \rho_y > 0$), la velocità critica oltre la quale il sistema diventa aperiodicamente instabile è:
$$u_{\text{crit}} = \sqrt{\frac{L}{-K}} = \frac{1}{\sqrt{\rho_y}}$$

---

## 7. Modello Ammortizzatore Non-Lineare Öhlins TTX25 MkII FSAE (`damper_model_ttx25.py`)

Implementa il modello matematico digressivo $C^\infty$ derivato direttamente dalle curve di banco dinamometrico ufficiali Öhlins TTX25 MkII per Formula Student:

### 7.1 Architettura & 4 Regolazioni Indipendenti (4-Way)
- **6 Specifiche di Valvolatura**: `C11/R11`, `C12/R12` (baseline FSAE di fabbrica), `C13/R13`, `C21/R21`, `C22/R22`, `C23/R23`.
- **Regolazioni Low-Speed (Bleed Orifice)**:
  - $LSC, LSR \in [0, 25]\text{ click}$ contati da tutto chiuso (stiff).
  - Modello esponenziale: $c_{ls}(\text{click}) = c_{ls,\min} + (c_{ls,\max} - c_{ls,\min}) e^{-\text{click}/\tau_{\text{click}}}$ con $\tau_{\text{click}} = 7.5$.
- **Regolazioni High-Speed (Shim Preload)**:
  - $HSC, HSR \in [0.0, 4.3]\text{ giri}$ contati da tutto aperto (soft).
  - Modello di rigidezza: $c_{hs}(\text{giri}) = c_{hs,\text{base}} + c_{hs,\text{gain}} \cdot \text{giri}$.

### 7.2 Formulazione Matematica Continua & Differenziabile ($C^\infty$)
Per garantire la convergenza simbolica in CasADi ed eliminare discontinuità di derivata al ginocchio di transizione ($v_{\text{knee}} \approx 0.022 \div 0.025\text{ m/s}$):
$$F_{\text{comp}}(v) = c_{ls,c} v_{k,c} \tanh\left(\frac{v}{v_{k,c}}\right) + c_{hs,c} \left( v - v_{k,c} \tanh\left(\frac{v}{v_{k,c}}\right) \right)$$
$$F_{\text{reb}}(v) = c_{ls,r} v_{k,r} \tanh\left(\frac{v}{v_{k,r}}\right) + c_{hs,r} \left( v - v_{k,r} \tanh\left(\frac{v}{v_{k,r}}\right) \right)$$
$$F_{\text{total}}(v) = \text{blend}(v) F_{\text{comp}}(v) + (1 - \text{blend}(v)) F_{\text{reb}}(v), \quad \text{blend}(v) = \frac{1}{2}(1 + \tanh(k_b v))$$

### 7.3 Trasformazione al Mozzo & Architettura Dual-Mode
- **Trasformazione Ruota**: $v_{\text{damper}} = MR \cdot v_{\text{wheel}}$, $F_{\text{wheel}} = MR \cdot F_{\text{damper}}$.
- **Dual-Mode Fallback**: L'utente può commutare liberamente tra la modalità `linear` (che utilizza $C_{\phi,f}, C_{\phi,r}, C_{z,f}, C_{z,r}$ costanti) e la modalità `ttx25_nonlinear` (che computa i momenti istantanei di smorzamento rollio e beccheggio in funzione delle velocità cinematiche delle ruote).

---

## 8. Direct Collocation Trajectory & Racing Line Optimizer (`src/mdo/mdo_direct_collocation.py`)

Modulo per la risoluzione del Problema di Controllo Ottimo (OCP) per la traiettoria a tempo minimo (*Minimum Lap Time & Optimal Racing Line*) lungo tracciati continui (Autocross, Skidpad, Endurance o tracciati custom da coordinate GPS/CAD).

### 8.1 Coordinate Curvilinee del Tracciato & Stato Trasformato
Il moto del veicolo è proiettato lungo la linea di mezzeria del tracciato parametrizzata dall'ascissa curvilinea $s \in [0, S_{\text{tot}}]$ e curvatura continua $\kappa(s)$:
- $s$: Coordinata indipendente di avanzamento lungo il centro pista $[m]$.
- $n(s)$: Scostamento laterale ortogonale alla mezzeria ($n > 0$ verso sinistra) $[m]$.
- $\xi(s) = \psi(s) - \theta(s)$: Orientamento relativo del telaio rispetto alla tangente del tracciato $[rad]$.
- $E(s) = \frac{1}{2} v(s)^2$: **Energia cinetica specifica per unità di massa** $[m^2/s^2]$. L'adozione di $E$ anziché della velocità scalare $v$ convexifica e linearizza la relazione tra accelerazione longitudinale $a_x$ e derivata spaziale $\frac{dE}{ds} = a_x$, eliminando singolarità numeriche a basse velocità ($\frac{dv}{ds} = \frac{a_x}{v} \to \infty$ per $v \to 0$).
- $a_x(s)$: Controllo di accelerazione/decelerazione longitudinale lungo l'asse vettura $[m/s^2]$.
- $\delta(s)$: Controllo dell'angolo di sterzata dell'assale anteriore $[rad]$.

### 8.2 Precondizionamento Numerico & Scaling $O(1)$
Per scongiurare il malcondizionamento delle condizioni di Karush-Kuhn-Tucker (KKT) e velocizzare la convergenza dell'algoritmo a punti interni IPOPT con MUMPS, tutte le variabili di decisione e i vincoli sono scalati e non-dimensionalizzati nell'intervallo compatto $[-1, 1]$:
$$\bar{n}_k = \frac{n_k}{n_{\text{scale}}}, \quad \bar{\xi}_k = \frac{\xi_k}{\xi_{\text{scale}}}, \quad \bar{E}_k = \frac{E_k}{E_{\text{scale}}}, \quad \bar{a}_{x,k} = \frac{a_{x,k}}{a_{x,\text{scale}}}, \quad \bar{\delta}_k = \frac{\delta_k}{\delta_{\text{scale}}}$$
dove $n_{\text{scale}} = \max\left(\frac{w_{\text{track}} - w_{\text{veh}}}{2}, 0.5\right)$, $\xi_{\text{scale}} = 42^\circ$, $E_{\text{scale}} = \frac{1}{2}(45\text{ m/s})^2$, $a_{x,\text{scale}} = \mu_x g$, $\delta_{\text{scale}} = 28^\circ$.

### 8.3 Funzione Obiettivo Multitermine & Allineamento MATLAB (`controlloModello_OCP_FSAE_1.m`)
La funzione obiettivo $J$ riproduce fedelmente l'architettura a quadratura multi-termine del file di riferimento MATLAB `controlloModello_OCP_FSAE_1.m`:
$$J = w_{\text{time}} \frac{T_{\text{lap}}}{T_{\text{scale}}} + J_{du} + J_{d^2u} + J_u + J_{dy} + J_{d^2y} + J_{\text{state}} + J_{\text{mayer}}$$

dove:
1. **Running Cost Lap Time (Lagrange Quadrature)**:
   $$T_{\text{lap}} = \sum_{k=0}^{N-2} \Delta t_k, \quad \Delta t_k = \frac{\Delta s \cdot (1 - n_{\text{mid},k} \kappa_{\text{mid},k})}{v_{\text{mid},k} \cos\xi_{\text{mid},k}}$$
   normalizzato rispetto al fattore di scala temporale $T_{\text{scale}}$ ($J_s$ nel modello MATLAB).
2. **Penalità Derivata Prima dei Controlli (Slew Rate - `duk` / `duk_t`)**:
   $$J_{du} = \sum_{k=0}^{N-2} \left[ r_{du,ax} \left(\frac{\Delta \bar{a}_{x,k}}{\Delta s}\right)^2 + r_{du,\delta} \left(\frac{\Delta \bar{\delta}_k}{\Delta s}\right)^2 \right]$$
   oppure in frequenza temporale $\Delta u / \Delta t_k$ impostando `use_time_rates=True`.
3. **Penalità Derivata Seconda dei Controlli (Control Curvature / Jerk - `duk2`)**:
   $$J_{d^2u} = \sum_{k=0}^{N-2} \left[ r_{du2,ax} (duk2_{ax,k})^2 + r_{du2,\delta} (duk2_{\delta,k})^2 \right]$$
   con $duk2_k = duk_{k+1} - duk_k$. Elimina rigorosamente le oscillazioni ad alta frequenza e il bang-bang chattering dello sterzo.
4. **Penalità Control Effort (`ru .* Uk`)**:
   $$J_u = \sum_{k=0}^{N-1} \left[ r_{u,ax} \bar{a}_{x,k}^2 + r_{u,\delta} \bar{\delta}_k^2 \right]$$
5. **Regolarizzazione Variazione Stati Ausiliari/Laterali (`rdy` & `rdy2`)**:
   $$J_{dy} + J_{d^2y} = \sum_{k=0}^{N-2} \left[ r_{dy,n} \left(\frac{\Delta \bar{n}_k}{\Delta s}\right)^2 + r_{dy2,n} (d2n_k)^2 + r_{dy,\xi} \left(\frac{\Delta \bar{\xi}_k}{\Delta s}\right)^2 + r_{dy2,\xi} (d2\xi_k)^2 \right]$$
6. **Centratura Traiettoria & Residui Terminali Mayer (Opzionali)**:
   $$J_{\text{state}} = \sum_{k=0}^{N-1} \left( r_n \bar{n}_k^2 + r_\xi \bar{\xi}_k^2 \right), \quad J_{\text{mayer}} = w_{\text{mayer},n} \bar{n}_{N-1}^2 + w_{\text{mayer},\xi} \bar{\xi}_{N-1}^2$$

Tutti i parametri di peso sono configurabili tramite la dataclass `DirectCollocationCostWeights` o dizionario equivalente passato al solutore `solve_direct_collocation_racing_line(cost_weights=...)`.


### 8.4 Equazioni di Collocazione Diretta (Metodo del Punto Medio)
I vincoli differenziali di stato sono integrati con schema trapezoidale implicito su ogni intervallo $[s_k, s_{k+1}]$ di ampiezza $\Delta s$:
$$\frac{n_{k+1} - n_k}{\Delta s} = \tan(\xi_{\text{mid},k}) \cdot (1 - n_{\text{mid},k} \kappa_{\text{mid},k})$$
$$\frac{\xi_{k+1} - \xi_k}{\Delta s} = \frac{\delta_k}{L} \cdot \frac{1 - n_{\text{mid},k} \kappa_{\text{mid},k}}{\cos(\xi_{\text{mid},k})} - \kappa_{\text{mid},k}$$
$$\frac{E_{k+1} - E_k}{\Delta s} = a_{x,k} \cdot \frac{1 - n_{\text{mid},k} \kappa_{\text{mid},k}}{\cos(\xi_{\text{mid},k})}$$

### 8.5 Vincoli Fisici del Veicolo & Bordo Pista
1. **Corridoio Stradale Permesso**:
   $$-n_{\max} \le n_k \le n_{\max}, \quad n_{\max} = \frac{w_{\text{track}} - w_{\text{veh}}}{2} - \text{margin}$$
2. **Ellisse d'Aderenza Combinata con Aerodinamica**:
   $$F_{z,\text{tot},k} = m g + \frac{1}{2}\rho S C_z (2 E_k)$$
   $$\mu_{y,\text{eff}} = \mu_y \cdot \frac{F_{z,\text{tot},k}}{m g}, \quad \mu_{x,\text{eff}} = \mu_x \cdot \frac{F_{z,\text{tot},k}}{m g}$$
   $$a_{y,\text{req},k} = 2 E_k \cdot \frac{\delta_k}{L}$$
   $$\left(\frac{a_{x,k}}{\mu_{x,\text{eff}} g}\right)^2 + \left(\frac{a_{y,\text{req},k}}{\mu_{y,\text{eff}} g}\right)^2 \le 1.0$$
3. **Limiti Powertrain Elettrico (Regole FSAE)**:
   $$F_{\text{trac},k} = m a_{x,k} + \frac{1}{2}\rho S C_d (2 E_k)$$
   $$F_{\text{trac},k} \le F_{x,\max,\text{torque}} = \frac{n_{\text{motors}} T_{\max,\text{motor}} \tau_{\text{final}} \eta_{\text{drive}}}{R_w}$$
   $$P_{\text{acc},k} = F_{\text{trac},k} \cdot \sqrt{2 E_k} \le P_{\max,\text{kw}} \cdot 1000 \quad (80\text{ kW})$$

### 8.6 Profilatore di Velocità Quasi-Statico Backward-Forward
Come warm-start per l'NLP e per valutazioni di telemetria istantanee (<5 ms), il modulo integra un profiler quasi-statico a 3 passaggi:
1. **Apex Speed Limit**: calcola per ogni punto la velocità limite pura in curva considerando l'equilibrio aerodinamico-centripeto iterativo:
   $$v_{\text{corner},i} = \sqrt{\frac{\mu_y (m g + \frac{1}{2}\rho S C_z v_{\text{corner},i}^2)}{m |\kappa_i|}}$$
2. **Backward Braking Pass**: retro-propaga i vincoli di decelerazione massima frenante a ritroso dai punti di corda (apex):
   $$v_{\text{back},i} = \min\left(v_{\text{corner},i}, \; \sqrt{v_{\text{back},i+1}^2 + 2 a_{\text{brake},\max} \Delta s}\right)$$
3. **Forward Acceleration Pass**: propaga in avanti la trazione utile considerando la curva di coppia del motore elettrico, la resistenza aerodinamica $F_{\text{drag}}$ e il tetto di 80 kW dell'accumulatore:
   $$v_{\text{fwd},i} = \min\left(v_{\text{back},i}, \; \sqrt{v_{\text{fwd},i-1}^2 + 2 a_{\text{trac},\max} \Delta s}\right)$$

### 8.7 Interfaccia Grafica Modulare & Disaccoppiamento Canvas (`src/ui/ui_direct_collocation.py`)
Per eliminare accoppiamenti rigidi e dipendenze circolari con il modulo del simulatore, l'interfaccia di Direct Collocation è strutturata in un componente autonomo e riutilizzabile:
- **Disaccoppiamento Canvas**:
  - `CollocationTrajectoryCanvas`: Canvas Matplotlib dedicato esclusivamente al rendering di traiettoria e controlli con 4 sotto-grafici sincroni (Racing Line vs Mezzeria con limiti nastro d'asfalto, Profilo di velocità $v(s)$ con confronto quasi-statico vs ottimo IPOPT, Inviluppo Controlli $a_x(s)$ e $\delta(s)$, Ellisse d'Aderenza normalizzata $a_x/(\mu g)$ vs $a_y/(\mu g)$).
- **Simulink Data Inspector Multi-Canale (`DataInspectorCanvas`)**:
  - Layout griglia flessibile a selezione dinamica: **1x1 (Singolo)**, **2x1 (Doppio)**, **3x1 (Triplo)** e **2x2 (Quadruplo)**.
  - Canali fisici selezionabili tramite checkbox: coordinate globali ($x, y$), velocità $v$, accelerazione tangenziale $a_x$, angolo di sterzo $\delta$, scostamento laterale $n$, orientamento relativo $\xi$, curvatura mezzeria $\kappa$, tempo progressivo $t$, accelerazione centripeta $a_y$, e canali derivati calcolati al volo (accelerazione totale $a_{tot} = \sqrt{a_x^2 + a_y^2}$, delta velocità $\Delta v = v - v_{\text{midpoint}}$, velocità d'imbardata geometrica $\dot{\psi} = v (\kappa + \dot{\xi})$, potenza longitudinale $P_{long} = m a_x v$).
  - Asse X commutabile in tempo reale tra ascissa curvilinea $s$ [m] e tempo sul giro $t$ [s].
  - Cursori verticali interattivi sincronizzati (*crosshair*): muovendo il mouse su uno qualsiasi dei subplot, una linea verticale tratteggiata si sposta in sincrono su tutti i grafici visualizzando i valori numerici esatti dei canali all'ascissa indicata.
- **Pannello Configurazione Tracciato (`TrackConfigPanel`)**:
  - Selezione rapida da libreria preset Formula SAE (*FSAE Autocross*, *FSAE Skidpad*, *FSAE Endurance*, *FSAE Acceleration*) o caricamento diretto da file `.csv` o `.json`.
  - Campi numerici e slider per larghezza nastro $w_{track}$ [m], discretizzazione $\Delta s$ [m], condizioni al contorno (velocità iniziale/finale, scostamento iniziale/finale, flag circuito chiuso ad anello).
  - Canvas di anteprima geometrica isolato con disegno istantaneo della mezzeria, nodi discretizzati e frecce direzionali.
- **Integrazione Design System**:
  - Piena conformità con `ui_theme_engine.py` (palette colori industriale scuro, font `Segoe UI`, zero emoji, indicatori di stato ASCII/badge `[OK]`, `[RUN]`, `[ERR]`).
  - Thread worker asincrono `CollocationWorker` (`QThread`) con feedback in tempo reale sul log di convergenza IPOPT e barra di avanzamento percentuale.

---

## 9. Simulatore di Guida Interattivo 14-DOF (`simulator.py` & `interactive_driver_14dof.py`)

L'ambiente di simulazione real-time 14-DOF è strutturato in un'architettura modulare industriale con entry point standalone (`simulator.py`) e facciata di retrocompatibilità (`interactive_driver_14dof.py`), suddiviso nei package dedicati:
- **`simulator.py`**: Entry point rapido e standalone con caricamento CLI (`--setup`, `--tir`), privo di dipendenze MDO per un boot istantaneo (<0.5 s).
- **`core/physics_thread.py`**: Thread fisico RK4 a 250/333/500 Hz (`PhysicsThread14DOF`), dinamica 14-DOF, trazione dual-motor / LSD / TV, degradazione termica pneumatici (`TireDegradationModel`), jacking kinematics e raycast collision mesh.
- **`core/gamepad_manager.py`**: Gestione input universale (`GamepadManager`, `FFB_PRESETS`, `XINPUT_VIBRATION_EX`), force feedback DirectInput/SDL2, trigger aptici per freno/trazione.
- **`ui/sim_viewport.py`**: Viewport OpenGL 3D ad alte prestazioni (`OpenGLViewport3D`, `_LegacyViewport3D`), telecamere multiple (*Chase, Cockpit POV, Nose, Bumper, 2D Top-Down*), rendering tracciato 3D con altimetria e micro-rugosità asfalto, ombre AO, segni di frenata e ghost car.
- **`ui/sim_main_window.py`**: Finestra principale di guida interattiva (`Interactive14DOFDrivingWindow`), selezione frequenze (RK4 e modello termico 10..100 Hz / sincrono), pannelli KPI e controlli assetto live.
- **`ui/sim_telemetry.py` & `ui/sim_controls.py`**: Finestre di telemetria avanzata multi-giro, esportazione dati e editor tracciati/profili stradali.

Consente di guidare il modello 14-DOF in tempo reale con hardware reale o tastiera:
- **Architettura Multi-Thread & Thread Safety**:
  - `PhysicsThread14DOF` (frequenza configurabile: **250 Hz** default / **333 Hz** / **500 Hz**, $\Delta t = 0.004\dots 0.002\text{ s}$): integra l'ODE non lineare con Runge-Kutta 4 (RK4). L'uso dei 250 Hz ottimizza i consumi CPU prevenendo qualsiasi rallentamento in tempo reale.
  - **Thread-Safe Snapshot Buffer**: Accesso concorrente alla cronologia di telemetria protetto da `threading.Lock()` (`get_telemetry_snapshot()`), eliminando race conditions e crash durante l'apertura simultanea della finestra grafici o esportazione CSV.
  - **Global Crash Handler (`setup_global_exception_handler`)**: Intercetta qualsiasi eccezione non gestita, salva il traceback completo in `crash_log.txt` e notifica l'utente con una finestra modale di errore esplicito.
- **Accelerazione C Nativa & Signature Cache (`ode_14dof_{sig}.dll`)**:
  - Compilazione nativa C con MinGW-GCC / MSVC dell'ODE 14-DOF (< 0.11 ms per step RK4 a 4 stadi, > 8800 Hz).
  - Cache deterministica basata su firma hash MD5 di `(VehicleParams, TireParams)` con sidecar `.sig`: azzera le collisioni di locking DLL su Windows e garantisce che la DLL caricata corrisponda esattamente alla combinazione attiva di veicolo e pneumatico.
  - Fallback trasparente a CasADi simbolico (`get_casadi_ode()`) in soli 21 ms di build (> 5100 Hz, 4.9% CPU): assicura la piena dinamicità su qualunque file `.tir` caricato e su qualsiasi macchina senza compilatore C.
- **Sistema Avanzato di Force Feedback & Feedback Aptico a 4 Canali (`GamepadManager`)**:
  - **Self-Aligning Torque (SAT) & Drop-Off al Limite**: Calcolo fisico della coppia di allineamento $T_{sw}$ (MF6.1) con braccio pneumatico $t_p$ e meccanico $t_m$; alleggerimento naturale al limite di aderenza anteriore.
  - **Grilletto Freno (LT - Impianto Idraulico Puro, NO ABS)**: Shudder meccanico ad alta aderenza e vibrazione da bloccaggio ruota / flatspotting ($\kappa < -10\%$) proporzionale alla severità della frenata.
  - **Grilletto Gas (RT - Trazione & Wheelspin)**: Chattering ad alta frequenza (32 Hz) in caso di pattinamento e perdita di trazione al retrotreno ($\kappa > +8\%$).
  - **Rumble Telaio (Bassa Frequenza)**: Sovrasterzo snap, sideslip rate $\dot{\beta}$, impatto finecorsa tamponi (bumpstops $z \to \text{gap}$) e fuori pista erba/ghiaia.
  - **Rumble Gomme & Texture Stradale (Alta Frequenza)**: Micro-vibrazione continua proporzionale a $v_x$, carico gomme anteriori, scrub chatter a limite aderenza e battito cordoli (kerbs 30–60 Hz).
  - Auto-rilevamento Windows Gaming Input (WGI), Xbox XInput nativo, PlayStation / Logitech DirectInput e tastiera.
- **Cruscotto Racing Digitale & Fullscreen HUD**:
  - Tachimetro / tachimetro ad arco ad alto contrasto con display digitale della velocità in km/h, barre dei pedali freno e acceleratore (con allarme lockup).
  - Banner di stato dinamico aderenza ([OK], [BLOCCAGGIO FRENI], [SOTTOSTERZO], [SOVRASTERZO], [TRAZIONE]).
  - Matrice 4 ruote con carichi verticali $F_z$, grip %, campanatura e **corsa sospensioni $z$ in mm rispetto allo zero statico a 1g**.
  - Cerchio di frizione G-G con vettore d'inerzia e coppie FFB istantanee.
  - Supporto Fullscreen nativo (tasto `F11` / pulsante dedicato).
- **Rendering Visivo 3D & 2D Corretto**:
  - Proiezione prospettica destrorsa con cinematica di sterzata coerente: la rotazione delle ruote anteriori segue fedelmente l'angolo di sterzo $\delta$ (destra $\rightarrow$ destra, sinistra $\rightarrow$ sinistra) sia nella vista 3D a inseguimento che nella mappa 2D top-down con gradiente di aderenza pneumatici.
- **Suite di Telemetria 14-DOF Completa**: 6 grafici sincronizzati per escursioni sospensioni, campanatura/convergenza dinamica, carichi verticali $F_z$ & jacking, diagramma G-G con colormap velocità, canali aptici/rumble e input pilota.

---

## 10. Mappe Cinematiche vs Wheel Travel & Curve Digitizer (`curve_digitizer.py`)

La suite include un sistema integrato per la gestione e digitalizzazione delle curve caratteristiche delle sospensioni:
- **Parametrizzazione SI a Standard 14-DOF**: Tutte le curve di sospensione (`camber_gain_f/r`, `toe_gain_f/r`, `rollcenter_f/r`) sono parametrizzate rispetto alla **corsa ruota verticale $z$ in metri [m]** (con $+z$ = compressione / bump, $-z$ = estensione / rebound).
- **Auto-Scaling SI nel Curve Digitizer**:
  Il modulo `curve_digitizer.py` permette di digitalizzare grafici di sospensione da PDF o immagini con coordinate X in millimetri [mm] o metri [m] e coordinate Y in gradi [°] o millimetri [mm]. Il tool esegue automaticamente il cambio di scala dei coefficienti polinomiali di 3° grado:
  $$y(z_m) = (10^9 c_3) z_m^3 + (10^6 c_2) z_m^2 + (1000 c_1) z_m + c_0$$
  garantendo compatibilità istantanea al 100% con il modello `Vehicle14DOF`.
- **Visualizzatore Sub-Tab 5 (Setup Tools)**:
  Permette di visualizzare simultaneamente in 4 subplot interattivi:
  1. Campanatura dinamica $\gamma(z)$ vs Corsa Ruota [mm] (Ant. vs Post.)
  2. Bump-Steer $\delta_{toe}(z)$ vs Corsa Ruota [mm] (Ant. vs Post.)
  3. Altezza Centri di Rollio $h_{rc}(z)$ vs Heave [mm]
  4. Geometria di Ackermann $\Delta\delta_{ack}$ vs Angolo Sterzo $\delta$ [°]
  Con possibilità di modificare istantaneamente i gain lineari o caricare file `.json` generati dal Curve Digitizer.

---

## 11. Finestre Pop-Out Standalone & Fullscreen (`PopoutPlotDialog`)

Ogni grafico e canvas della suite può essere aperto in una finestra standalone separata e indipendente:
- **Clonazione Vettoriale Figure**: L'istanza Matplotlib viene clonata tramite serializzazione veloce (`pickle.dumps`/`loads`), preservando l'integrità del renderer principale e consentendo pan/zoom/export indipendenti.
- **Funzionalità Finestra Pop-Out**:
  - Modalità a tutto schermo tramite tasto rapido **F11** o pulsante dedicato.
  - Toolbar completa di navigazione Matplotlib (Pan, Zoom, Reset, Subplot adjust).
  - Esportazione ad alta risoluzione (PNG vettoriale a 300 DPI o PDF/SVG).

---

## 12. Corsa Reale Ammortizzatori, Tamponi Finecorsa & Molle Installate

Integrazione della fisica dei finecorsa e delle molle installate a bordo vettura:
- **Modello Non-Lineare Tampone Finecorsa (`eval_bumpstop_force`)**:
  - Compressione libera fino al traferro $z \le z_{\text{gap}}$.
  - Ingaggio progressivo con legge elastica non lineare:
    $$F_{\text{bs}} = K_{\text{bs}} \cdot \Delta z + 50 K_{\text{bs}} \cdot \Delta z^2$$
  - Arresto rigido se viene superata la corsa fisica massima dell'ammortizzatore ($z \ge z_{\text{bump,max}}$).
- **Altezze da Terra Dinamiche & Bottoming Check (`eval_dynamic_ground_clearance`)**:
  - Calcolo istantaneo di $h_{\text{ground}}(t) = h_{\text{static}} - z(t) - \text{deflection}$. Rilevamento automatico spanciata / chassis grounding ($h \le 0$).
- **Sizing Molle e Barre Antirollio Esteso**:
  - Confronto immediato tra frequenze target di Flat Ride e frequenze reali installate $f_{n,\text{inst}}$.
  - Calcolo del cedimento statico a 1g ($z_{\text{sag}}$) e del margine libero prima del tampone (*Bump Headroom*).

---

## 13. Manuale Utente & Guida Dinamica Veicolo (`docs/USER_GUIDE.md`)

Manuale di 10 capitoli integrato all'interno dell'applicazione (Menu Bar `Help & Guides` $\rightarrow$ `📖 Manuale Utente & Guida Dinamica del Veicolo`):
- Tratta la teoria analitica completa (modelli stazionari, YMD, Pacejka MF6.1, principi di assetto), le modalità di utilizzo della suite, il workflow del Digitizer e un cheatsheet di assetto per la pista.

---

## 14. Finestra di Bilanciamento Ottimale & Sweep Incertezza Parametrica (`mdo_balance_range.py`)

Modulo per la determinazione della **Finestra di Bilanciamento Ottimale** (*Optimal Robust Balance Corridor*) su sweep di parametri geometrici ed elastici con iniezione simultanea di incertezze multi-sorgente:
- **Parametri di Sweep Supportati**:
  - `LLTD` (Ripartizione Trasferimento Carico Laterale Ant. $0.35 \dots 0.75$)
  - `wd_f` (Distribuzione Masse Avantreno $0.40 \dots 0.60$)
  - `aero_balance` (Bilanciamento Aerodinamico $0.35 \dots 0.60$)
  - `rc_height_f` / `rc_height_r` (Altezze Centri di Rollio $0 \dots 150\text{ mm}$)
  - `h_cg` (Altezza Baricentro $0.20 \dots 0.38\text{ m}$)
  - `K_phi_f` / `K_phi_r` (Rigidezza di Rollio Molle/ARB)
- **Iniezione Incertezze Parametriche Worst-Case**:
  - Aderenza Pneumatici: $\pm \Delta\mu\%$ (tolleranza asfalto/temperatura)
  - Distribuzione Masse: $\pm \Delta WD_f\%$ (spostamento baricentro/pilota)
  - Migrazione Aerodinamica: $\pm \Delta Aero_f\%$ (beccheggio in frenata/curva)
  - Rigidezze/ARB: $\pm \Delta LLTD\%$
- **Criteri di Ottimalità & Robustezza**:
  - Sottosterzo Nominale: $K_{us,\text{nom}} \in [0.10, 0.80]^\circ/\text{g}$ a velocità tipica ($12.0\text{ m/s}$).
  - Robustezza Worst-Case: $K_{us,\text{worst}} > 0.0^\circ/\text{g}$ (zero rischio sovrasterzo incontrollato).
  - Margine Alta Velocità: $LLTD - Aero \ge +1.5\% \dots +2.0\%$ (prevenzione snap-oversteer).
  - Punteggio di Robustezza (*Robust Balance Score* $0\dots 100\%$) e visualizzazione bivariata con twin-axis ($K_{us}$ vs $(LLTD-Aero)\%$ e $a_{y,\text{max}}$).

---

## 14. Yaw Moment Diagram (YMD) & Handling Diagram (`src/mdo/mdo_target_optimization.py`, `src/utils/vehicle_dna_maps.py`)

Modulo per l'esplorazione esaustiva del diagramma momento d'imbardata - accelerazione laterale $(a_y, M_z)$ o coefficienti adimensionali $(C_y, C_n)$ secondo la metodologia dello Yaw Moment Diagram (YMD), integrato con il Diagramma di Handling e il metodo di continuazione numerica.

### 14.1 Il Diagramma del Momento d'Imbardata (Yaw Moment Diagram - YMD)
Il diagramma mappa l'intero spazio di manovrabilità del veicolo su una griglia bivariata di input controllabili dal pilota o dall'assetto:
- **Angolo di Sterzo Ruota Anteriore**: $\delta \in [-\delta_{\max}, \delta_{\max}]$ (curve iso-steer $\delta = \text{cost}$).
- **Angolo di Assetto Baricentrico del Telaio**: $\beta = \arctan(v_y / v_x) \in [-\beta_{\max}, \beta_{\max}]$ (curve iso-sideslip $\beta = \text{cost}$).
- A velocità di avanzamento assegnata $v_x$, per ogni combinazione $(\delta, \beta)$ il modello valuta:
  1. La cinematica di sterzatura differenziale di Ackermann e l'orientamento delle 4 ruote.
  2. L'angolo di deriva di ciascun pneumatico:
     $$\alpha_{fl} = \delta_{fl} - \arctan\left(\frac{v_y + a r}{v_x - \frac{t_f}{2} r}\right), \quad \alpha_{fr} = \delta_{fr} - \arctan\left(\frac{v_y + a r}{v_x + \frac{t_f}{2} r}\right)$$
     $$\alpha_{rl} = -\arctan\left(\frac{v_y - b r}{v_x - \frac{t_r}{2} r}\right), \quad \alpha_{rr} = -\arctan\left(\frac{v_y - b r}{v_x + \frac{t_r}{2} r}\right)$$
  3. Il carico verticale $F_{z,i}$ risultante da peso statico, carico aerodinamico $F_{\text{down}} = \frac{1}{2}\rho S C_z v_x^2$ e trasferimento di carico laterale rollio-dipendente:
     $$\Delta F_{z,f} = \frac{K_{\phi,f} \phi + m_f a_y q_1}{t_f}, \quad \Delta F_{z,r} = \frac{K_{\phi,r} \phi + m_r a_y q_2}{t_r}$$
  4. Le forze laterali $F_{y,i}$ e longitudinali $F_{x,i}$ generate tramite il modello non-lineare Pacejka MF6.1 accoppiato alla campanatura dinamica $\gamma_i$ e al camber thrust.
  5. La risultante di accelerazione laterale d'inerzia $a_y$ e il momento d'imbardata netto $M_z$:
     $$a_y = \frac{1}{m}\left[ (F_{y,fl} + F_{y,fr})\cos\delta + (F_{x,fl} + F_{x,fr})\sin\delta + F_{y,rl} + F_{y,rr} \right]$$
     $$M_z = a (F_{y,fl} + F_{y,fr})\cos\delta - b (F_{y,rl} + F_{y,rr}) + \frac{t_f}{2}(F_{x,fr} - F_{x,fl})\cos\delta + \frac{t_r}{2}(F_{x,rr} - F_{x,rl})$$

### 14.2 Indicatori Fisici Chiave & Derivate di Stabilità
Dal diagramma YMD vengono estratti automaticamente i parametri fondamentali di handling:
1. **Asse di Trim Stazionario ($M_z = 0$)**:
   Rappresenta tutti i punti in cui il veicolo è in equilibrio d'imbardata in curva stabilizzata. L'intercetta di massima accelerazione laterale lungo l'asse $M_z = 0$ identifica il grip laterale massimo stazionario $a_{y,\text{max,trim}}$.
2. **Autorità di Controllo dello Sterzo ($N_\delta$)**:
   $$\left.N_\delta(v_x) = \frac{\partial M_z}{\partial \delta}\right|_{\delta=0, \beta=0} \quad [\text{Nm/deg}]$$
   Quantifica la prontezza del veicolo nel generare momento d'imbardata all'azionamento del volante. Deve essere strettamente positivo ($N_\delta > 0$).
3. **Stabilità Direzionale Statica ($N_\beta$)**:
   $$\left.N_\beta(v_x) = \frac{\partial M_z}{\partial \beta}\right|_{\delta=0, \beta=0} \quad [\text{Nm/deg}]$$
   Quantifica l'effetto stabilizzante di banderuola generato dall'asse posteriore in presenza di un disturbo di sbandamento laterale. Per un veicolo intrinsecamente stabile a comandi bloccati deve essere negativo ($N_\beta < 0$).
4. **Margine di Controllo (*Control Margin* $CM$)**:
   $$CM(v_x) = \left|\frac{N_\delta(v_x)}{N_\beta(v_x)}\right|$$
   Un rapporto elevato indica un avantreno molto autoritario e reattivo; un rapporto basso denota un retrotreno dominante con forte tendenza sottosterzante.
5. **Margine di Riserva di Momento ($\Delta M_z$)**:
   Distanza verticale tra l'asse di trim $M_z = 0$ e il contorno limite dell'inviluppo YMD a una data accelerazione $a_y$. Definisce il momento d'imbardata massimo ancora disponibile per il pilota per correggere un sovrasterzo improvviso o forzare l'imbardata in inserimento.

### 14.3 Linee Iso-Raggio di Curvatura ($R = \text{cost}$)
Il modulo traccia linee verticali di riferimento centripeto corrispondenti alle curve caratteristiche del regolamento FSAE e dei circuiti di gara:
$$a_{y,\text{req}}(R) = \frac{v_x^2}{g \cdot R}$$
- $R = 9.125\text{ m}$: Raggio geometrico mezzeria Skidpad ufficiale Formula SAE / Formula Student.
- $R = 15.0\text{ m}$: Tornante stretto (Hairpin / tornante lento Autocross).
- $R = 30.0\text{ m}$: Variante a media velocità (Chicane / esse veloce).
- $R = 50.0\text{ m}$: Curva veloce di raccordo.
- $R = 100.0\text{ m}$: Curva veloce da pieno carico aerodinamico.
L'intersezione tra la retta $a_y = a_{y,\text{req}}(R)$ e l'asse $M_z = 0$ indica l'angolo di sterzo $\delta$ e di assetto $\beta$ richiesti per percorrere quella specifica curva alla velocità $v_x$. Se la linea iso-raggio cade oltre il bordo dell'inviluppo YMD, la curva non è fisicamente percorribile a quella velocità (perdita di controllo/fuori pista).

### 14.4 Diagramma di Handling & Convenzione di Segno del Gradiente
Il Diagramma di Handling rappresenta la relazione tra l'accelerazione laterale $a_y$ e la variabile cinematica di sterzatura $\left(\frac{L}{R} - \delta\right)$ o la curva dello sterzo al trim stazionario $\delta(a_y)$.  
Conformemente alla definizione classica del gradiente di sottosterzo:
$$\rho_y = -\frac{K}{L} \iff \mathbf{K = -L \cdot \rho_y}$$
- **Condizione Necessaria di Stabilità Stazionaria**:
  $$\rho_y < \frac{1}{u^2} \iff -\frac{K}{L} < \frac{1}{u^2}$$
- **Risoluzione con Continuazione Pseudo-Arc-Length**:
  I metodi numerici convenzionali di Newton falliscono in prossimità del picco di aderenza laterale ($\frac{da_y}{d\delta} \to 0$, matrice Jacobiana singolare). Il modulo implementa un algoritmo di continuazione pseudo-arc-length che parametrizza la curva tramite la lunghezza d'arco $s$:
  $$\mathbf{F}(\mathbf{x}(s), \lambda(s)) = \mathbf{0}, \quad \left\|\frac{d\mathbf{x}}{ds}\right\|^2 + \left(\frac{d\lambda}{ds}\right)^2 = 1$$
  consentendo di seguire la curva di handling senza discontinuità oltre il limite di aderenza, mappando la transizione verso il sovrasterzo terminale (*spin*) o sottosterzo terminale (*push*).

---

## 15. Sistema di Telemetria Avanzata 14-DOF & Layout Personalizzabili

Il modulo di telemetria del simulatore di guida 14-DOF (`FullTelemetryWindow`) include:
- **23 Canali Fisici Modulari & Load Case Generator**:
  1. Corsa Sospensioni $z$ [mm] (FL, FR, RL, RR)
  2. Campanatura Dinamica $\gamma$ [°] (FL, FR, RL, RR)
  3. Convergenza Dinamica $\delta_{\text{toe}}$ [°] (FL, FR, RL, RR)
  4. Carichi Verticali $F_z$ [N] & Jacking Caster/KPI $\Delta F_z$ [N]
  5. Forze Laterali Gomme $F_y$ [N]
  6. Angoli di Deriva Pneumatici $\alpha$ [°]
  7. Rapporti di Slittamento Longitudinale $\kappa$ [%]
  8. Percentuale di Utilizzo Aderenza [%]
  9. **Diagramma G-G ($a_y$ vs $a_x$) Isotropo**: Rapporto di aspetto 1:1 rigorosamente quadrato (`set_aspect('equal')`), limiti simmetrici `[-2.5g, +2.5g]`, cerchi di attrito concentrici a $1.0\text{g}, 1.5\text{g}, 1.8\text{g}, 2.0\text{g}$, filtraggio delle fasi di fermo ($v_x < 2\text{ km/h}$) e rimozione di spike numerici da derivate ODE.
  10. Assetto Cassa (Rollio $\phi$, Beccheggio $\theta$, Velocità d'Imbardata $r$)
  11. Velocità Chassis ($v_x$, $v_y$, Angolo di Deriva Baricentrico $\beta$)
  12. Input Pilota (Sterzo $\delta$, Gas %, Freno %)
  13. Feedback Aptico & Rumble (% Grilletti LT/RT, Rumble Telaio/Gomme)
  14. Centri di Rollio Dinamici $h_{rc,f}, h_{rc,r}$ [mm] (altezza e migrazione laterale $y_{rc}$)
  15. Temperature Gomme Bulk [°C] (FL, FR, RL, RR)
  16. Strati Termici 3-Zone (Superficie, Battistrada, Carcassa) [°C]
  17. Usura Gomme [%] & Moltiplicatore Grip
  18. **Coppie Ruote (Trazione e Frenata) [Nm]** ($T_{\text{drv}}$, $T_{\text{brk}}$ per mozzo)
  19. **Forze Longitudinali Gomme $F_x$ [N]** (MF6.1 combinata)
  20-23. **Terna Forze Ruota Triassiali ($F_x, F_y, F_z, |F|$) [N] per FL, FR, RL, RR** (Generatore di carichi per dimensionamento FEA braccetti, portamozzi e pushrod)
- **Preset e Layout Personalizzati (`CustomTelemetryLayoutDialog`)**:
  - 6 Preset predefiniti per analisi rapida (Setup, Gomme, Geometria/Assetto, Guida/Pilota, Termica/Degrado, Carichi Sospensioni & FEA).
  - Configuratore per creare layout su misura (1, 2, 4, 6 o 8 grafici) assegnando qualsiasi canale a qualsiasi slot.
- **Focus Singolo Grafico, Zoom Fluido & Popout Fullscreen Standalone (`TelemetryPopoutDialog`)**:
  - Menu di selezione rapida per ingrandire qualsiasi grafico a piena risoluzione.
  - Tasto **F11 / Schermo Intero** sia per la finestra principale sia per i singoli grafici aperti in pop-out indipendente.
  - **Zoom Fluido con Rotella del Mouse**: ascoltatore eventi `scroll_event` per zoomare continuamente centrati sul puntatore del mouse lungo gli assi del tempo e del diagramma G-G, con pulsanti di reset rapido (`🔍 Reset Zoom`, `↔ 10s`, `↔ 30s`).

---

## 16. Sistema di Gestione Layout Tracciati (AutoX, Skidpad, Endurance & Custom Track Loader)

Il modulo `mdo_direct_collocation.py` e il simulatore di guida interattivo `interactive_driver_14dof.py` supportano la selezione e l'importazione di geometrie tracciato arbitrarie:

### Preset Integrati:
1. **FSAE AutoX (850 m)**: Slalom a 6 coni, chicanes ad alto trasferimento di carico, tornante a raggio decrescente, curvone veloce ad alto carico aerodinamico ed esse tecnica.
2. **FSAE Skidpad (Figure-8)**: Configurazione standard a doppio anello concentrico con raggi $R = 9.125\text{ m}$ (diametro interno $15.25\text{ m}$, corsia $3.0\text{ m}$).
3. **FSAE Endurance (1250 m)**: Circuito completo da $1.25\text{ km}$ con staccate profonde, allunghi ad alta velocità e sequenze complesse.
4. **FSAE Acceleration (200 m)**: Rettilineo continuo con zona di frenata e arresto finale.

### Formati di File Tracciato Supportati (`tracks/*.csv`, `tracks/*.json`):
- **CSV**: Colonne `s,x,y,theta,kappa,w_left,w_right` o coordinate spline grezze `x,y,w_left,w_right` interpolate automaticamente con B-Spline periodiche.
- **JSON**: Struttura dizionario contenente gli array `s`, `x_center`, `y_center`, `theta`, `kappa`, `w_left`, `w_right`.
- **Funzioni di I/O**: `load_custom_track_file(file_path)` e `save_track_file(file_path, track_dict)` per esportare e importare layout da software esterni (ad es. OptimumLap, ChassisSim, Bosch Data Logger).

---

## 17. Selezione Modelli Fisici, Visualizzazione Parametri ("ALL") & Valutazione a Parametri Costanti

Per garantire rigore fisico e massima flessibilità operativa:
- **Disaccoppiamento tra Modello Fisico e Visualizzazione Parametri**:
  - Il selettore principale **Vehicle Model** (`self.model_combo`) seleziona esclusivamente il modello fisico attivo per tutte le simulazioni e calcoli (`3-DOF Steady-State MDO`, `7-DOF Transient (Tier 2)`, `14-DOF Full Vehicle (Tier 2)`).
  - All'interno della tab *Optimizer Params*, il selettore **Vista Parametri** (`self.combo_param_view`) consente di scegliere tra:
    1. `📌 Parametri Modello Attivo`: visualizza esclusivamente i parametri pertinenti alla fisica correntemente selezionata.
    2. `📋 Mostra Tutti i Parametri (ALL)`: visualizza e rende modificabili tutte le **13 categorie fisiche** (oltre 40 parametri: geometria, masse, aerodinamica, powertrain, sospensioni, barre ARB, elastocinematica, rigidezze, mappe non-lineari).
- **Valutazione su Parametri Costanti ("Run Constant Parameters")**:
  - Aggiunto il pulsante dedicato `⚡ Valuta Parametri Costanti (No Ottimizzazione)` sotto a *Run Global Optimization*.
  - Esegue la valutazione istantanea del veicolo ai parametri impostati (senza eseguire l'ottimizzazione SLSQP): genera simultaneamente lo **Yaw Moment Diagram (YMD)** con il modello fisico attivo, l'**Handling Diagram di Pacejka**, il **diagramma 3D G-G-V**, le **MAPs (Map of Achievable Performance)**, la **KPI Dashboard**, la stima della coppia al volante, il dimensionamento sospensioni e le mappe cinematiche.
- **Supporto Multi-Modello YMD (3-DOF, 7-DOF, 14-DOF)**:
  - Il calcolo dello Yaw Moment Diagram (`generate_mmm_diagram` e `generate_multispeed_mmm_envelopes`) rispetta fedelmente la formulazione dello Yaw Moment Diagram ([`documents/YMD.pdf`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/documents/YMD.pdf)): per ogni coppia $(\delta, \beta)$ a velocità $v_x$, risolve l'equilibrio della forza laterale $\Sigma F_y - m v_x r = 0$ ed estrae $a_y$ e il momento d'imbardata non bilanciato $M_z$.
  - Il calcolo integra automaticamente la campanatura e convergenza statiche nel 7-DOF, e le dinamiche di rollio, escursioni sospensioni, rigidezze e camber gain polinomiali nel 14-DOF.
- **Barra di Gestione Profili Setup (`setups/*.json`)**:
  - Dropdown con i profili salvati (`default_vehicle.json`, `fsae_autocross_setup.json`, `fsae_skidpad_setup.json`, `high_downforce_setup.json`).
  - **💾 Salva**: sovrascrive direttamente le modifiche sul file di parametri attivo.
  - **💾 Salva con Nome...**: crea un nuovo file `.json` e lo aggiunge all'elenco dei profili.
  - **📂 Sfoglia...**: carica qualsiasi file JSON di setup esterno aggiornando istantaneamente l'interfaccia.
  - Serializzazione/Deserializzazione completa gestita tramite `VehicleParams.to_dict()`, `VehicleParams.from_dict()`, `VehicleParams.save_json()`, `VehicleParams.load_json()`.

---

## 18. Simulatore di Guida 14-DOF Hardware-Accelerato & Fisica Real-Time

Il modulo simulatore interattivo [`interactive_driver_14dof.py`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/interactive_driver_14dof.py) è stato potenziato con un'architettura ad alte prestazioni:

### 18.1 Fisica Real-Time & C-Code Generation CasADi
- **Time Accumulator Sub-Stepping**: Risolto il collo di bottiglia del timer di Windows (risoluzione 15.6 ms di `time.sleep`). Il ciclo fisico adotta un pattern ad accumulatore temporale con sub-stepping dinamico (fino a 8 sotto-passi RK4 per ciclo) e cooperative micro-sleeping a precisione sub-millisecondo, garantendo un *Real-Time Ratio* $1.00\pm0.01$ costante a 250 Hz, 333 Hz e 500 Hz.
- **C-Code Generation CasADi & Compilazione Nativa Ottimizzata**: La classe [`Vehicle14DOF`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/vehicle_14dof.py) include i metodi `compile_c_ode()`, `load_compiled_ode()` e `get_fast_ode()`. L'ODE a 14 gradi di libertà (23 stati) viene esportata in C (`ode_14dof_gen.c`), compilata con flag di massima ottimizzazione (`gcc -shared -fPIC -O3 -ffast-math` o MSVC `/LD /O2 /fp:fast`) in libreria dinamica a 64-bit (`ode_14dof_gen.dll`) e caricata via `ca.external`.
  - **Rilevamento Automatico Compilatore**: Il sistema cerca in ordine: `cl` / `gcc` nel PATH, compilatore portabile locale `tools/mingw/bin/gcc.exe` (MinGW-w64 / w64devkit), percorsi standard MSYS2/UCRT (`C:\msys64\ucrt64\bin\gcc.exe`, etc.) e Visual Studio tramite `vswhere.exe`.
  - **Resilienza Path Unicode su Windows**: Il caricamento tramite `ca.external` gestisce i percorsi Windows con caratteri non-ASCII (es. cartella `Università`) risolvendo automaticamente l'8.3 short path via `GetShortPathNameW`, eliminando gli errori di caricamento dynamic library WIN32 Error 126.
  - **Prestazioni Verificate**: Singola valutazione ODE C nativa ridotta a **~19 µs** (< 0.02 ms), con tempo di integrazione RK4 completo (4 valutazioni + aggiornamenti di stato) di **~0.11 ms/step** (corrispondente a oltre **8500 Hz** di throughput teorico). Se nessun compilatore è presente, il motore ricade automaticamente e senza eccezioni sull'ODE simbolica CasADi `MXFunction`.

### 18.2 Rendering 3D Hardware-Accelerato & Vettura Volumetrica con `QOpenGLWidget`
- **Migrazione Strutturale**: La classe [`OpenGLViewport3D`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/interactive_driver_14dof.py) sfrutta la pipeline OpenGL con Z-Buffer hardware:
  - **Asfalto Poligonale & Zero Z-Fighting**: Asfalto ardesia a strisce continue con `glPolygonOffset` e linea di mezzeria tratteggiata anti-flicker ad altezza differenziata e cordoli bicolore (rosso/bianco a sinistra, giallo/rosso a destra).
  - **Coni FSAE 3D**: Troncoconici tridimensionali (blu a sinistra, arancioni a destra) disposti geometricamente lungo i margini pista.
  - **Vettura Formula Student Volumetrica 3D**:
    - **Monoscocca & Musetto**: Telaio solido con cockpit scavato, musetto rastremato e crash box anteriore in carbonio.
    - **Pance Laterali (Sidepods)**: Prese d'aria radiatori 3D con condotti sagomati.
    - **Cockpit, Casco & Halo in Titanio**: Casco pilota oro/giallo con visiera scura specchiata, struttura di sicurezza Halo in titanio e roll hoop posteriore.
    - **Ali Aerodinamiche 3D**: Ala anteriore multi-elemento con flap e paratie verticali rosse; ala posteriore ad alto carico con profili alari 3D e piloni a collo di cigno (*swan-neck*).
    - **Ruote Cilindriche 3D & Sospensioni a Doppio Braccio Oscillante**: Pneumatici volumetrici a 16 segmenti con battistrada e cerchi in lega a stella a 5 razze; braccetti wishbone (superiore, inferiore, pushrod diagonale e tirante sterzo) animati in tempo reale in funzione dell'escursione verticale $z_{fl}, z_{fr}, z_{rl}, z_{rr}$.
    - **Colorazione Grip & Skid Marks Smooth con Alpha Blending**: Cerchi/pinze con colorazione dinamica in base al grip (Verde $<60\%$, Giallo $60-88\%$, Rosso $>88\%$) e **strisce di gommatura continue** (*skid ribbons*) con trasparenza alpha e dissolvenza naturale temporale.
  - **Mini HUD Overlay**: Velocità in km/h, cronometro giro e bolla G-Force dinamica.
  - **Fallback**: Mantenuta la classe `_LegacyViewport3D` con fallback automatico.

### 18.3 Modalità Fullscreen Esclusiva di Gioco & Preset Telecamera
- **Game-Only Fullscreen**: Premendo **`F11`**, **`Esc`** o il pulsante **`⛶ Schermo Intero (F11)`**, il pannello laterale dei controlli viene collassato e il viewport 3D di guida si espande al **100% dello schermo** con HUD da gara integrato (velocità, lap timer, G-meter, e badge flottante di uscita rapida).
- **5 Preset Telecamera (Tasti 1–5)**:
  1. `1`: **Chase Cam** (inseguimento standard a volo d'uccello).
  2. `2`: **Cockpit / Halo POV** (visuale soggettiva dal casco del pilota).
  3. `3`: **Nose Cam** (camera anteriore montata sul musetto).
  4. `4`: **Drone / High Cam** (visuale aerea panoramica).
  5. `5`: **2D Top-Down** (mappa pista bidimensionale).
- **Pannello Regolazioni Live & Zoom**: Slider dedicati per FOV (300-800), Distanza (1-15 m), Altezza (0.3-8 m) e Smorzamento; zoom continuo tramite rotella del mouse.

### 18.4 Editor & Disegnatore Tracciato 2D Interattivo (`TrackDesignerDialog`)
- **Canvas Interattivo a Nodi e Spline**: Permette di disegnare e modificare tracciati con click del mouse (aggiunta, spostamento drag & drop ed eliminazione nodi).
- **Generazione Spline Cubica**: Interpolazione automatica per circuiti chiusi ad anello o piste sprint aperte, con calcolo in tempo reale della larghezza pista, cordoli e mezzeria.
- **Libreria Preset**: Include template per *FSAE Autocross*, *FSAE Skidpad (Figure-8)*, *Circuito Tecnico*, *Slalom Test* e *Anello Ovale*.
- **Analisi & Integrazione**: Calcolo istantaneo di lunghezza totale ($m$), raggio minimo di curvatura ($R_{min}$ in $m$), numero di curve, esportazione/importazione su file CSV (`tracks/*.csv`) e pulsante **`🏁 Carica nel Simulatore`** per guidare istantaneamente sul tracciato disegnato.

### 18.5 Revisione Avanzata Simulatore: Lap Timer, Degrado Gomme, Grip Customizzabile & Anti-Flicker
- **Cronometro Lap Timer Reale con Fotocellula**:
  - Il cronometro non è più un semplice tempo trascorso, ma rileva il transito fisico attraverso il gate di partenza/arrivo (`S/F Photocell Line`) posizionato al primo punto del tracciato.
  - Supporta la **partenza da fermo sulla griglia** (il veicolo viene posizionato 2.5 m dietro la fotocellula; il cronometro si attiva al passaggio della linea).
  - Memorizza e visualizza nell'HUD fullscreen e nella sidebar: numero giro (`LAP N`), tempo giro attuale (`00:00.00`), ultimo giro (`LAST`), miglior tempo (`BEST`) e **delta live vs. best** colorato dinamicamente (verde = miglioramento, rosso = peggioramento).
- **Arresto Completo del Veicolo (0.0 km/h) & Standstill Lock**:
  - Rimossa la saturazione minima a 0.05 m/s. Il veicolo può decelerare fino a 0.0 km/h tramite il freno. Quando la velocità scende sotto 0.20 m/s con freno attivo e senza acceleratore, si attiva lo *Standstill Lock*, bloccando le velocità planari e le rotazioni ruota a zero ed evitando micro-deriva numerica.
- **Risoluzione Bug Auto-Sterzatura (Focus Loss & Deadzone)**:
  - Implementati gli eventi `focusOutEvent` e `changeEvent` in `Interactive14DOFDrivingWindow`: quando la finestra perde il focus (Alt-Tab, minimizzazione o click su dialoghi esterni), lo stato dei tasti (`key_states`) viene azzerato istantaneamente, evitando bloccaggi dello sterzo a fine corsa.
  - Aumentata la deadzone di default del gamepad a 0.08 (8%) per evitare drift di assi non centrati.
- **Rendering Pista Anti-Flicker per Figure-8 (Skidpad)**:
  - Eliminato il windowing ristretto basato su singolo `argmin` che causava violenti sfarfallii all'intersezione dello Skidpad e delle curve ravvicinate. La pista viene renderizzata per intero con continuità geometrica.
  - Nei circuiti chiusi ad anello, l'ultimo punto viene agganciato al primo senza discontinuità (*loop closure*).
  - Disegnata la linea di partenza/traguardo a scacchi (*checkered S/F line*) con torrette di cronometraggio a fotocellula illuminate in ciano ai bordi pista.
- **Fattore di Grip Pista Customizzabile & Superfici Dinamiche**:
  - Slider dedicato nella sidebar da 10% a 150% con preset rapidi: **Asciutto (1.0)**, **Umido (0.75)**, **Bagnato (0.50)** e **Ghiaccio (0.15)**.
  - Rilevamento dinamico di uscita di pista (*off-track*, grip ridotto al 35%) e transito sui cordoli (*kerbs*, grip all'85%).
  - Parametro $L_{MUX}$ reso dinamico e parametrico nel modello a 14 gradi di libertà, scalato in sincrono con $L_{MUY}$.
- **Modello di Degrado Termico ed Usura Pneumatici ([`tire_degradation.py`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/tire_degradation.py))**:
  - Calibrato sui dati sperimentali TTC (Tire Test Consortium) dello pneumatico **Hoosier 43075 16x7.5-10 R20** (Rim 7, Run 4; riferimento in `tire_datas/mu_vs_temp.png`).
  - **Curva Termica Parabolica**: Modello termico parabolico calcolato sul 95° percentile dei dati di picco a diversi bin di temperatura:
    $$\mu_y(T) = \mu_{peak} - a \cdot (T - T_{opt})^2$$
    con $T_{opt} = 56.2^\circ\text{C}$, $\mu_{peak} = 1.6918$, $a = 0.00088331\text{ } (^\circ\text{C})^{-2}$.
  - **Fattore di Aderenza Normalizzato**:
    $$f_{temp}(T) = \max\left( \text{floor}, 1.0 - c_{temp} \cdot (T - T_{opt})^2 \right)$$
    con $c_{temp} = a / \mu_{peak} = 0.00052211\text{ } (^\circ\text{C})^{-2}$ e soglia di tenuta minima $\text{floor} = 0.80$ (per evitare divergenze quadratiche a freddo o in forte surriscaldamento). Nel range misurato ($44^\circ\text{C} - 70^\circ\text{C}$):
    - A $44^\circ\text{C}$: $f_{temp} = 0.922$ ($\mu_y \approx 1.56$)
    - A $50^\circ\text{C}$: $f_{temp} = 0.980$ ($\mu_y \approx 1.66$)
    - A $56.2^\circ\text{C}$: $f_{temp} = 1.000$ ($\mu_y \approx 1.692$, picco assoluto)
    - A $60^\circ\text{C}$: $f_{temp} = 0.993$ ($\mu_y \approx 1.68$)
    - A $70^\circ\text{C}$: $f_{temp} = 0.901$ ($\mu_y \approx 1.52$)
  - Modalità selezionabili live: `Disattivato`, `Leggero (1-Zona Bulk)` e `Dettagliato (3-Zone: Battistrada, Bulk, Carcassa)`.
  - Riscaldamento basato sulla potenza di slittamento istantanea $P_{slip} = |F \cdot v_{slip}|$ e raffreddamento convettivo verso l'ambiente.
  - Modificatore di aderenza $\mu_{eff} = \mu_{base} \cdot f(T) \cdot f(W)$ integrato nel loop fisico e monitorato nell'HUD 4 ruote con temperatura e percentuale di usura per ciascun corner.

### 18.6 Feedback Aptico Grilletti, Selezione Giri, Esportazione Dati & Cockpit POV
- **Feedback Aptico Grilletti (Impulse Triggers LT/RT)**:
  - Supporto nativo hardware per Microsoft Xbox Impulse Triggers tramite chiamata non documentata XInput Ordinal 100 (`XInputSetStateEx` con struct `XINPUT_VIBRATION_EX`) e wrapper SDL2 `SDL_GameControllerRumbleTriggers`.
  - Architettura a doppia attuazione (*dual actuation*): i micro-motori fisici dei grilletti (LT freno e RT gas) operano a 16-bit completi (0-65535) senza tagli; in aggiunta, una frazione proporzionata viene miscelata ai motori palmari sinistro e destro, garantendo un feedback tattile netto anche su controller standard o connessi via Bluetooth standard Windows.
  - Caratterizzazione frequenziale delle vibrazioni:
    * **LT (Freno / Bloccaggio Ruote)**: pulsazione quadra da ABS pump stutter a 16 Hz, con ampiezza modulata sulla severità del bloccaggio ($\kappa < -12\%$) o alto utilizzo aderenza ($util > 85\%$).
    * **RT (Acceleratore / Trazione)**: ronzio rapido a denti di sega a 38 Hz, attivo su pattinamento ruote posteriori in accelerazione ($\kappa > 12\%$) e burnout da fermo sulla linea di partenza.
  - Controlli utente dedicati nella sidebar: slider "Sensibilità Aptica Grilletti" e pulsante di test istantaneo `"⚡ Prova Feedback Grilletti (LT/RT)"`.
- **Selezione Manuale Giri in Telemetria (`LapSelectionDialog`)**:
  - Finestra di selezione interattiva richiamabile dal pulsante `"Seleziona Giri..."` con checkbox per ogni giro registrato nella sessione, corredati di tempo sul giro ($s$), distanza percorsa ($m$), velocità media ($km/h$) e contrassegno ★ per il miglior giro.
  - Modalità overlay selettivo (`"Confronto Giri Selezionati (Overlay)"`): visualizza e confronta simultaneamente solo i giri scelti dall'utente sia sul diagramma G-G che sui 16 canali fisici.
  - Pulsanti di selezione rapida: `Tutti`, `Deseleziona`, `Best vs Ultimo`, `Inverti`.
- **Esportazione Dati di Telemetria Multi-Formato**:
  - Menu dedicato `"💾 Esporta Dati..."` integrato nella toolbar superiore di `FullTelemetryWindow`:
    1. **`Esporta CSV Completo (Tutti i Giri)`**: Genera file CSV conforme agli standard MoTeC / MATLAB con tutte le 50+ variabili sincrone nel tempo e nella distanza.
    2. **`Esporta CSV Giri Selezionati (Overlay MoTeC)`**: Concatena ed esporta unicamente i giri contrassegnati nella selezione attiva per confronto multi-lap.
    3. **`Esporta JSON Dati Sessione`**: Dump strutturato completo di metadati, statistiche giro, layout e campioni grezzi.
    4. **`Salva Grafico Attuale (PNG / PDF 300 DPI)`**: Esportazione raster o vettoriale ad alta risoluzione del layout telemetrico corrente.
- **Visuale Prima Persona / Cockpit (POV Pilota)**:
  - Telecamera posizionata esattamente agli occhi del pilota ($x_{eye} = x_{car} + 0.10\cos\psi$, $y_{eye} = y_{car} + 0.10\sin\psi$, $z_{eye} = z_{car} + 0.56\text{ m}$), orientata in avanti a 35 metri lungo beccheggio e imbardata con accoppiamento dinamico al rollio laterale $\phi_{car}$ nei cordoli e in curva.
  - Rendering abitacolo dedicato: disattivazione automatica del casco pilota in visuale cockpit per evitare occlusioni interne; visibilità del telaio monoscocca Formula Student, volante/yoke rotante animato con lo sterzo $\delta$, triangoli di sospensione e ruote anteriori sterzanti.
  - Assegnazione rapida al **Tasto 2** della tastiera, integrazione nel selettore a tendina e ciclo telecamere sequenziale con il **Tasto V**.

### 18.7 Camera Smoothing in Prima Persona & Live-Tuning Parametri Veicolo
- **Camera Smoothing Continuo per Visuali Onboard (`COCKPIT`, `NOSE`, `BUMPER`)**:
  - Elimina il tremolio ad alta frequenza (*jitter*) causato dalla discretizzazione numerica dell'assetto 14-DOF e dalle brusche variazioni di pendenza altimetrica e cordoli.
  - Implementazione con filtro esponenziale continuo tempo-invariante:
    $$\mathbf{p}_{\text{cam}}(t) = \mathbf{p}_{\text{cam}}(t - \Delta t) + \left(1 - e^{-\lambda \Delta t}\right) \left(\mathbf{p}_{\text{target}}(t) - \mathbf{p}_{\text{cam}}(t - \Delta t)\right)$$
    con costanti temporali calibrate:
    * Posizione planare $xy$: $\lambda_{pos} = 18.0\text{ s}^{-1}$
    * Quota altimetrica $z$: $\lambda_z = 10.0\text{ s}^{-1}$ (smorzamento potenziato delle asperità e salti di pendenza)
    * Vettore di puntamento (*look-at target*): $\lambda_{rot} = 22.0\text{ s}^{-1}$
    * Beccheggio e rollio: $\lambda_{pitch} = 12.0\text{ s}^{-1}$
  - Ortonormalizzazione sferica del forward vector e dell'up vector con vincolo d'inclinazione per prevenire divergenze e gimbal lock in frenata o compressione.
- **Finestra Esterna Fluttuante: Live Vehicle Parameters Inspector & Tuning (`LiveVehicleParametersInspector`)**:
  - Finestra autonoma non modale (`Qt.Window`, aperta con `.show()`), che non interrompe né congela il ciclo di simulazione OpenGL o l'integrazione a 250 Hz di `PhysicsThread14DOF`.
  - **Tab Ispezione Telemetria Live**: Aggiornamento a 25 Hz dei parametri e degli stati dinamici del veicolo (velocità, accelerazioni $a_x, a_y$, corse e carichi sospensioni $F_{z,i}$, temperature pneumatici e grip residuo).
  - **Tab Live-Tuning Parametri Veicolo**:
    * Sospensioni & Barre: Rigidezza molle ant/post $K_{s,f}, K_{s,r}$ [N/m], barre antirollio $K_{\phi,arb,f}, K_{\phi,arb,r}$ [N/m], rigidezza bump stop.
    * Smorzamento: Clic ammortizzatori Öhlins TTX25 MkII Low-Speed ($LSC, LSR$) anteriori e posteriori con hot-reload istantaneo delle curve digressive $C^\infty$.
    * Aerodinamica & Masse: Coefficienti di deportanza e drag $C_z, C_d$, area frontale $S$, ripartizione aero $\text{bal}_{aero}$, massa pilota/zavorra $m$, bilanciamento longitudinale $bal_{w,f}$.
    * Pneumatici & Grip: Coefficienti di picco $\mu_x, \mu_y$, pressioni di gonfiaggio ant/post $P_f, P_r$ [bar].
  - **Funzionalità Operative**:
    * Checkbox **`[x] Auto-Apply modifiche istantanee`**: applica le variazioni al motore fisico al rilascio di ogni slider o modifica numerica.
    * Pulsante **`⚡ Applica a Caldo (Live Hot-Reload)`**: inietta i parametri istantaneamente nel simulatore tramite `apply_tuned_params_to_thread(params_dict)`.
    * Pulsante **`↺ Ripristina Valori Nominali`**: ripristina istantaneamente il setup nominale di fabbrica (`nominal_fsae_params()`).
    * Pulsanti **`💾 Esporta Setup JSON`** e **`📂 Carica Setup JSON`**: persistenza delle configurazioni personalizzate.

---

## 19. Istruzioni per la Compilazione dell'Eseguibile Standalone

Per compilare la suite in un eseguibile Windows standalone (`.exe`) senza dipendenze Python esterne:

1. **Attiva l'ambiente virtuale**:
   ```bash
   .\venv\Scripts\activate
   ```
2. **Esegui la compilazione con PyInstaller**:
   ```bash
   .\venv\Scripts\pyinstaller.exe --noconfirm MDO_Target_Suite.spec
   ```
   *oppure tramite lo script batch:*
   ```cmd
   build_exe.bat
   ```
3. **Risultato della Build**:
   L'eseguibile completo si troverà in:
   `dist\MDO_Target_Suite\MDO_Target_Suite.exe`

### 18.7 Modifiche Rapide di Assetto In-Simulator & Sincronizzazione MDO Bidirezionale
- **Tab Dedicato "Assetto & Aerodinamica" nella Sidebar**:
  - **Geometria Statica degli Angoli Ruota**: Regolazione fine e continua con `QDoubleSpinBox` per campanatura statica anteriore/posteriore (`camber_f_static`, `camber_r_static` da -6.0° a +1.0°) e convergenza statica anteriore/posteriore (`toe_f_static`, `toe_r_static` da -3.0° a +3.0°). Le modifiche si riflettono istantaneamente sul vettore parametri `p_vec` dell'integrazione RK4 a 250 Hz senza interruzioni.
  - **Pressioni Pneumatici a Freddo (Cold Tire Pressures)**:
    - Regolazione indipendente anteriore e posteriore in bar con conversione live in psi ($1\text{ bar} = 14.5038\text{ psi}$).
    - Integrazione diretta con il modello non-lineare Pacejka MF6.1 (`tire_model_mf61.py`), che scala la rigidezza d'angolo $C_{F\alpha}$ e l'aderenza di picco $\mu_{peak}$ al variare della pressione reale di contatto.
    - Preset rapidi per 0.70 bar (10.1 psi), 0.85 bar (12.3 psi - Nominale ET18), 1.00 bar (14.5 psi).
  - **Assetto Sospensioni Decoppiato (Rigidezza K, Precarico, Puntone L0 & Ride Height)**:
    - Regolazione indipendente della rigidezza molla $K_{spring}$ ($20 - 150\text{ N/mm}$), del precarico $P$ ($0 - 1500\text{ N}$) e dell'offset lunghezza puntone cinematismo $L_0$ ($\pm 10\text{ mm}$, $2\text{ cm}$ totali).
    - Decoppiamento tra Altezza da Terra (Ride Height) e precarico: la variazione di altezza da terra regola il puntone $L_0$ preservando il precarico nominale senza forzare valori spuri.
    - Corsa geometrica totale teorica alla ruota $z_{wheel,tot} = \text{shock\_stroke} / MR = 57.0\text{ mm} / 0.8 = 71.25\text{ mm}$ (pienamente conforme al requisito FSAE $\ge 50\text{ mm}$ totali, senza vincoli spuri sul sag elastico a 1g).
    - Tampone bumpstop modellato con contatto progressivo negli ultimi $5\text{ mm}$ di corsa dello stelo ammortizzatore e impaccamento solido a $3.5\text{ mm}$ residui.
    - Badge tecnico dinamico che calcola in tempo reale carico statico $F_{z,stat}$, sag statico 1g, corsa bump pre-tampone, corsa bump totale e verifica FSAE.
  - **Configurazione Aerodinamica & Pacchetto Ali**:
    - Preset aerodinamici integrati:
      * **ET18 Baseline**: $C_z = 3.421$, $C_x = 1.200$, $\text{Aero Balance} = 50.9\%$ anteriore.
      * **Basso Carico / Low Drag**: $C_z = 1.850$, $C_x = 0.720$, $\text{Aero Balance} = 46.0\%$ anteriore.
      * **Carico Medio / Balanced**: $C_z = 2.600$, $C_x = 0.950$, $\text{Aero Balance} = 48.5\%$ anteriore.
      * **Personalizzato (Manuale)**: regolazione libera di $C_z$, $C_x$ e bilanciamento percentuale.
    - Tacche d'incidenza flap anteriore e posteriore (Tacche 1..4: $-15\%$, $-5\%$, Base, $+10\%$).
    - Badge tecnico aerodinamico con calcolo in tempo reale dell'efficienza aerodinamica $L/D = C_z / C_x$, ripartizione $\%$, carico deportante e resistenza a $60\text{ km/h}$ in Newton e chilogrammi equivalenti ($F_{down} / 9.81$).
  - **Link MDO Bidirezionale e Ripristino ET18**:
    - Segnale Qt `setup_changed = pyqtSignal(object)` emesso a ogni variazione di parametro. Sia `MainWindowPro` che `MainWindowClassic` intercettano il segnale e aggiornano i parametri dell'ambiente multidisciplinare MDO, sincronizzando tabelle di input e mappe cinematiche.
    - Pulsante dedicato `"Ripristina ET18 Baseline"` per resettare istantaneamente assetto e aerodinamica ai valori nominali di fabbrica.
    - Salvataggio e caricamento immediato dei setup in formato `.json` compatibile con l'intero ecosistema DTS.

### 18.8 Ricalibrazione Termica Gomme & Protezione Telemetria (Stint Recovery)
- **Ricalibrazione Modello Termico Pneumatici (`tire_degradation.py`)**:
  - Aumentata la massa termica effettiva del bulk battistrada da $1000\text{ J/K}$ a $2000\text{ J/K}$ per riflettere la reale inerzia termica di uno pneumatico FSAE da 10"/13".
  - Ridotto il coefficiente convettivo a $8.5\text{ W/K}$, estendendo la costante di tempo di raffreddamento a $\tau \approx 235\text{ s}$ (~4 minuti), eliminando il raffreddamento irrealisticamente repentino.
  - Modello a 3 zone con pesatura rappresentativa bilanciata: $T_{disp} = 0.35 T_{surface} + 0.50 T_{tread} + 0.15 T_{carcass}$.
- **Protezione Telemetria e Recupero Ultimo Stint**:
  - Buffer circolare `last_stint_telemetry` protetto da lock di concorrenza (`_history_lock`) che preserva l'intera sessione prima del reset.
  - Finestra di dialogo di avviso se si tenta di resettare la pista con più di 100 campioni di telemetria non salvati.
  - Pulsante `"Recupera Stint"` nella toolbar rapida per esportare in CSV i dati dell'ultimo stint azzerato in qualsiasi momento.
- **Minimappa HUD & Curva Sterzo Ackermann**:
  - Minimappa vettoriale 2D top-down renderizzata in overlay nell'angolo del viewport di guida per orientamento visivo immediato del pilota.
  - Angolo di sterzatura differenziale derivato dalla curva cinematica reale dell'angolo pignone (`pinion steer vs wheel steer`) specchiata simmetricamente rispetto all'asse y, garantendo che la ruota interna curvi fisicamente e visivamente più della ruota esterna.

### 18.9 Cinematica del Piano Medio Stradale 3D & Disaccoppiamento Real-Time 500 Hz
- **Risoluzione Anomalia Carico Verticale Nullo ($F_z = 0\text{ N}$) su Rialzi/Pendenze**:
  - *Diagnosi*: Nelle formulazioni precedenti basate sull'altezza media macroscopica $z_{\text{road,macro}} = \frac{1}{4}\sum z_{\text{road},i}$, su una rampa con pendenza del 5–10% la differenza di quota tra asse anteriore e posteriore ($\Delta z \approx \pm 40\text{ mm}$) superava di gran lunga la deflessione statica elastica dello pneumatico ($\delta_{0,i} \approx 8\text{ mm}$). Ne conseguiva una deflessione calcolata $\delta_i < 0$, inducendo un falso distacco ruota (*lift-off*) con carico verticale $F_z = 0\text{ N}$ da fermo.
  - *Formulazione Cinematica*: Il contatto suolo 3D viene ora proiettato sul **piano medio stradale locale**, calcolato dai 4 punti di contatto ruota:
    $$\theta_{\text{road}} = -\frac{(z_{\text{road},fl} + z_{\text{road},fr}) - (z_{\text{road},rl} + z_{\text{road},rr})}{2 L}, \quad \phi_{\text{road}} = \frac{(z_{\text{road},fl} + z_{\text{road},rl}) - (z_{\text{road},fr} + z_{\text{road},rr})}{2 t_m}$$
    $$z_{\text{plane},i} = z_{\text{mid}} \mp \theta_{\text{road}} \frac{L}{2} \pm \phi_{\text{road}} \frac{t_m}{2}$$
    $$\Delta z_{\text{micro},i} = z_{\text{road},i} - z_{\text{plane},i}, \quad \delta_i = \delta_{0,i} - z_{\text{chassis},i} + \Delta z_{\text{micro},i}$$
    dove $z_{\text{chassis},i}$ è valutato rispetto all'assetto relativo cassa-strada $\theta_{\text{rel}} = \theta - \theta_{\text{road}}$ e $\phi_{\text{rel}} = \phi - \phi_{\text{road}}$. Su qualsiasi pendenza o sopraelevazione uniforme, $\Delta z_{\text{micro},i} \equiv 0$ e $z_{\text{chassis},i} \to 0$, garantendo $F_{z,i} = F_{z,0,i} > 0\text{ N}$ continuo.
- **Disaccoppiamento Frequenze: 500 Hz Fisica / 60 Hz Rendering / 15 Hz Dashboard**:
  - *Physics Thread*: La chiamata ridondante a `self.ode()` per calcolare l'accelerazione $a_{x,\text{raw}}$ è stata eliminata riutilizzando direttamente il vettore derivate $\mathbf{k}_1[0]$. I calcoli computazionalmente pesanti (modello Pacejka MF6.1 a 4 ruote per forze combinate e auto-allineanti $M_z$, haptics e instanziazione dizionari telemetrici) sono stati confinati a 60 Hz (`target_emit_dt`). Nei tick intermedi, il thread esegue unicamente l'integrazione RK4 in ~0.8 ms, garantendo frequenza fisica stabile a 500 Hz senza jitter.
  - *GUI Event Loop*: Per azzerare la contesa del GIL Python tra thread grafico e thread numerico, il rendering OpenGL 3D (`sim_viewport.py`) continua a operare a 60 FPS fluidi, mentre l'aggiornamento testuale degli oltre 70 `QLabel`, progress bar e parsing CSS QSS (`setStyleSheet`) in `sim_main_window.py` è stato cadenzato a 15 Hz con caching dello stato, riducendo del 75% l'overhead sul thread principale Qt.
  - *Raycast 3D Ottimizzato*: Il raycasting del fondo pista in `TrackCollisionMesh.raycast_down` è stato riscritto con proiezione baricentrica 2D scalare diretta, riducendo il tempo di esecuzione di 28 volte (46 ms vs 1289 ms per 50'000 query).

### 18.10 Risoluzione Equilibrio Statico su Piste Sopraelevate (Banking) & Invalidazione DLL C
- **Diagnosi dell'Anomalia di Carico Asimmetrico da Fermo su Banking**:
  - *Sintomo*: Su tracciati con pendenza trasversale/sopraelevazione (es. banking a $3.8^\circ$), il veicolo da fermo mostrava carichi anomali `FL: 0N [LIFT-OFF]`, `RL: 0N [LIFT-OFF]`, `FR: 4297N (clamped a Fz_max)`, `z_fl = +49.1 mm` (a tampone) e `z_fr = -34.2 mm` (a finecorsa estensione).
  - *Cause Radice Risolte*:
    1. **Asimmetria Carreggiate ($t_f \ne t_r$) sul Piano Medio**: Nel prototipo FSAE ET-18, la carreggiata anteriore ($t_f = 1.252\text{ m}$) è maggiore di quella posteriore ($t_r = 1.136\text{ m}$). Su un piano stradale inclinato di angolo $\phi_{\text{road}}$, la quota tra i punti d'appoggio scala linearmente con la carreggiata locale: $\Delta z_f = \phi_{\text{road}} t_f$ all'anteriore e $\Delta z_r = \phi_{\text{road}} t_r$ al posteriore. L'adozione della cinematica a piano medio differenziato azzera i gradienti fittizi di svergolamento $\Delta z_{\text{micro},i} \equiv 0$, garantendo che la deflessione gomma coincida esattamente con il sag elastico nominale.
    2. **Momento di Rovesciamento Gravitazionale nel Riferimento Inerziale**: Poiché il vettore accelerazione di gravità $\mathbf{g} = [0, 0, -g]^T$ è fisso nel sistema di riferimento inerziale mondiale, il braccio della forza peso rispetto all'asse di rollio dipende dall'angolo di rollio assoluto $\phi$, e non dal rollio relativo $\phi_{\text{rel}}$. L'equazione di momento destabilizzante $M_{\text{roll,destab}} = m A_y h_{\text{arm}} + m g (h_{\text{cg}} - h_{\text{rc}}) \sin(\phi)$ ripristina la fisica newtoniana esatta.
    3. **Smorzamento Progressivo a Fermo (*Standstill Settling*)**: Nelle fasi di arresto prolungato, le velocità verticali dei mozzi $vz_i$ e il moto di sussulto $w_{\text{heave}}$ vengono smorzati esponenzialmente ($vz_i \leftarrow 0.85 \cdot vz_i$) anziché essere azzerati istantaneamente a ogni tick. Ciò consente alle masse non sospese e alla cassa di assestarsi naturalmente nella posizione di equilibrio elastico a 1g senza restare bloccate in posizioni deformate fuori asse.
    4. **Allineamento Iniziale in `reset_state()`**: All'avvio della sessione o dopo un reset pista, l'assetto angolare iniziale del veicolo $\phi$ e $\theta$ viene inizializzato con i valori macroscopici $bank(s_0)$ e $grade(s_0)$ della stazione di partenza, eliminando transitori a gradino e urti all'attivazione del solutore.
    5. **Invalidazione Automatica della Cache DLL C (`ode_14dof_gen.dll`)**: Per prevenire il riutilizzo silenzioso di binari compilati C obsoleti dopo modifiche alle equazioni differenziali ordinarie in `vehicle_14dof.py`, la funzione `get_model_signature()` incorpora ora un hash crittografico MD5 del file sorgente Python. Qualsiasi aggiornamento alle equazioni fisiche o ai coefficienti invalida la cache e forza la rigenerazione e ricompilazione immediata del modulo C a 64-bit (`ode_14dof_{sig}.dll`).
- **Validazione Numerica su Banking a 3.8°**:
  - Rollio stabilizzato a regime: $\phi = 3.86^\circ \approx \phi_{\text{road}}$.
  - Escursione mozzi a riposo: $|z_i| < 0.25\text{ mm}$ (static 1g ground equilibrium).
  - Carichi verticali ruota: $F_{z,fl} = 650\text{ N}$, $F_{z,fr} = 733\text{ N}$, $F_{z,rl} = 693\text{ N}$, $F_{z,rr} = 778\text{ N}$.
  - Carico totale $\Sigma F_z = 2853\text{ N} \approx m g \cos(3.8^\circ) = 2858\text{ N}$ con trasferimento di carico laterale naturale verso la ruota a valle di $\sim 41\text{ N}$ per ruota e zero falsi lift-off.

---

---

## 21. Analisi di Sensitività Parametrica & Simulatore Punteggi Formula Student 2026 (`mdo_sensitivity.py`)

Il modulo `mdo_sensitivity.py` introduce due strumenti fondamentali per l'ingegneria di pista e lo sviluppo veicolo in Formula Student:

### 21.1 Regolamento Ufficiale Punteggi FS-Rules 2026 (Section D 9 & Table 3, Table 11)
Implementa le formule ufficiali di attribuzione punteggio per le 5 discipline dinamiche:
- **Formula Generale Disciplina Standard**:
  $$\text{SCORE} = (P_{\max} - P_{\min}) \cdot \left( \frac{T_{\max} - T_{\text{team}}}{T_{\max} - T_{\min}} \right)^2 + P_{\min}$$
- **Specifiche Discipline (CV & EV Manual Mode)**:
  1. **Acceleration (75 m)**: $P_{\max} = 50.0\text{ pt}$, $P_{\min} = 2.5\text{ pt}$ ($0.05 P_{\max}$), $T_{\max} = 1.70 \cdot T_{\min}$.
  2. **Skidpad ($R=9.125\text{ m}$)**: $P_{\max} = 50.0\text{ pt}$, $P_{\min} = 2.5\text{ pt}$ ($0.05 P_{\max}$), $T_{\max} = 1.35 \cdot T_{\min}$.
  3. **Autocross (1 Giro)**: $P_{\max} = 100.0\text{ pt}$, $P_{\min} = 10.0\text{ pt}$ ($0.10 P_{\max}$), $T_{\max} = 1.40 \cdot T_{\min}$.
  4. **Endurance (22 km)**: $P_{\max} = 250.0\text{ pt}$, $P_{\min} = 25.0\text{ pt}$ ($0.10 P_{\max}$), $T_{\max} = 1.50 \cdot T_{\min}$.
  5. **Efficiency**: $P_{\max} = 75.0\text{ pt}$, con fattore di efficienza $EF = T_{\text{uncorr}}^2 \cdot E_{\text{kwh}}$ e formula:
     $$\text{SCORE}_{\text{eff}} = P_{\max} \cdot \left( \frac{EF_{\max} - EF_{\text{team}}}{EF_{\max} - EF_{\min}} \right)^2, \quad EF_{\max} = 2.0 \cdot EF_{\min}$$
- **Punteggio Dinamico Totale**: $\mathbf{525.0\text{ pt}}$ massimi.

### 21.2 Benchmark di Competitività da FS-World Ranking
La suite include i preset di tempi di riferimento $T_{\min}$ per i principali eventi internazionali (FS Germany / Hockenheimring, FS Austria / Red Bull Ring, FS Italy / Varano, FS East / Hungaroring) con possibilità di inserire benchmark personalizzati o calibrare su dati di gara reali.

### 21.3 Motore di Sensitività 1D & Diagramma Tornado / Pareto
- **Sweep 1D**: Variazione continua del parametro scelto (rapporto di trasmissione `final_drive`, massa `m`, distribuzione masse `wd_f`, aerodinamica, rigidezze rollio, altezze centri di rollio, camber, toe, grip $\mu$, potenza motore) con tracciamento della curva di risposta e marcatore del punto nominale di progetto.
- **Tornado / Pareto Multi-Parametro**: Variazione simultanea a $\pm 10\%$ su tutti i principali parametri di progetto e ordinamento automatico dal parametro con impatto maggiore a quello con impatto minore.

---

## 22. Generazione e Risoluzione Non-Lineare Esatta delle MAPs tramite Continuazione Arc-Length (`mdo_target_optimization.py`, `vehicle_dna_maps.py`)

La suite implementa la generazione e visualizzazione delle **Map of Achievable Performance (MAPs)** e delle **Mappe di Stabilità DNA ($n_1 \dots n_6$)** secondo la trattazione rigorosa delle prestazioni limite stazionarie e del notebook di riferimento Mathematica (`MAPs L17.nb`):

### 22.1 Solutore Non-Lineare a Continuazione Pseudo Arc-Length (Keller's Method)
Per garantire la convergenza numerica continua e l'assenza di salti o singolarità in prossimità della tangenza al limite di aderenza ($\frac{\partial a_y}{\partial \delta} = 0$), il solutore delle curve a velocità costante $u = \text{cost}$ risolve il sistema aumentato:
$$\begin{cases} 
\text{res}(u, \delta, \beta, r) = 0 \\ 
T_k^T (X - X_k) - \Delta s = 0 
\end{cases}, \quad X = (\beta, r, \delta)^T$$
- Il passo $\Delta s$ viene effettuato lungo il vettore tangente 3D $T_k$, garantendo la non-singolarità del Jacobiano su tutto il dominio di manovrabilità fino al punto di picco $a_{y,\max}(u)$ e sul ramo post-saturazione.
- Le curve a $u = \text{cost}$ generano una caratteristica di manovrabilità liscia e continua, consentendo la determinazione esatta dell'inviluppo di massima aderenza $a_{y,\max}(u)$ con guadagno aerodinamico quadratico ($F_{\text{down}} \propto u^2$).

### 22.2 Famiglie a Grandezze Costanti e Bordi di Realizzabilità
Le 6 mappe fondamentali sono costruite tramite l'intersezione di reticoli a curve di livello:
1. **Piano di Ingresso $(\delta, u)$**: Funnel di manovrabilità con sweep a $u=\text{cost}$ (fasci orizzontali), curve iso-$a_y$ (iperboli), curve iso-$R$ (parabole cinematiche da Ackermann $L/R$) e bordo limite $\delta_{\text{peak}}(u)$.
2. **Piano Misto $(u, a_y)$**: Curve a sterzo costante $\delta=\text{cost}$ intersecate da parabole di raggio costante $R=\text{cost}$ ($a_y = u^2/R$), con inviluppo superiore $a_{y,\max}(u)$.
3. **Piano di Trim $(\delta, a_y)$**: Curve caratteristiche di manovrabilità a $u=\text{cost}$ con bordo superiore al locus $\frac{\partial a_y}{\partial \delta} = 0$.
4. **Piano di Uscita / Spazio di Stato $(\rho, \beta)$**: Curve chiuse a $u=\text{cost}$ e ventaglio di livelli iso-$a_y$, delimitati dal limite cinematico $\beta = b\rho$ e dal bordo di saturazione dell'asse posteriore.
5. **Piano del Guadagno di Curvatura $(\delta, \rho)$**: Risposta $\rho(\delta)$ a $u=\text{cost}$ con iperboli iso-$a_y$ ($\rho = a_y/u^2$).
6. **Piano di Curvatura Massima $(u, \rho)$**: Inviluppo $\rho_{\max}(u) = a_{y,\max}(u)/u^2$ con curve a $\delta=\text{cost}$.

---

## 23. Geometria di Ackermann Differenziale, Singolarità a Zero Gradi & Regolarizzazione (`vehicle_14dof.py`, `params_vehicle.py`)

### 23.1 Formulazione Cinematica & Singolarità a $\delta \to 0$
Nei sistemi di sterzo a quadrilatero articolato (scatola a cremagliera e tiranti di sterzo), la percentuale di Ackermann rispetto alla geometria teorica a 100% è definita come:
$$\% \text{Ackermann}(\delta) = \frac{\Delta \delta_{\text{reale}}(\delta)}{\Delta \delta_{\text{ideale}}(\delta)} \times 100\%$$
dove la differenza geometrica ideale tra ruota interna ed esterna è ricavata dalla relazione di Ackermann pura:
$$\cot\delta_{o,\text{ideal}} - \cot\delta_{i,\text{ideal}} = \frac{t_f}{L} \implies \Delta \delta_{\text{ideale}}(\delta) \approx \frac{t_f}{L}\delta^2 + \mathcal{O}(\delta^4)$$
Notare che il denominatore $\Delta \delta_{\text{ideale}}$ è un infinitesimo del **secondo ordine** ($\propto \delta^2$).

#### Origine del Picco Asintotico a $\delta \to 0$ in Adams Car:
Nelle simulazioni multibody Adams Car (`11_ackermann_f__ackerman_percent_vs_steer_angle.jpeg`):
1. A volante dritto ($\delta \to 0$), sia il numeratore $\Delta \delta_{\text{reale}}$ che il denominatore $\Delta \delta_{\text{ideale}}$ tendono a zero (forma indeterminata $\frac{0}{0}$).
2. Nella realtà meccanica e nei solutori numerici a vincoli cinematici, è presente una convergenza statica nominale ($\delta_{\text{toe},0} \neq 0$) o una micro-asimmetria lineare/tolleranza di convergenza $c_1 \delta + \epsilon$ dell'algoritmo di Newton-Raphson.
3. Dividendo per $\delta^2$, per $\delta \to 0$ il rapporto diverge asintoticamente:
   $$\lim_{\delta \to 0} \frac{c_1 \delta + \epsilon + c_2 \delta^2}{\frac{t_f}{L}\delta^2} = \pm \infty$$
   Questo provoca l'impennata verticale del grafico (asintoto verso l'alto $> 120\%$) nella finestra ristretta $|\delta| < 6^\circ \dots 8^\circ$.

### 23.2 Risoluzione Matematica: Regolarizzazione Analitica di de L'Hôpital
La cinematica del quadrilatero di sterzo reale è $C^\infty$ (continua e derivabile infinite volte). Escludendo l'artefatto asintotico numerico di Adams Car ($|\delta| < 8^\circ$), la risposta fisica è perfettamente simmetrica rispetto allo zero ed è descritta con $R^2 > 0.999$ da un polinomio quadratico pari privo di singolarità:
$$\% \text{Ackermann}(\delta_{sw}) = a_2 \delta_{sw}^2 + a_0$$
Per $\delta \to 0$, applicando la regola di de L'Hôpital:
$$\lim_{\delta \to 0} \% \text{Ackermann}(\delta) = a_0$$
- **ET-18 (2026)**:
  - Polinomio: $\% \text{Ack}(\delta_{sw}) = 1.1009 \times 10^{-3} \cdot \delta_{sw}^2 + 50.985\%$
  - $a_0 = 50.99\%$ (51.0% nominale a riposo)
  - A fondo corsa ($\pm 125^\circ$ volante / $\pm 26.5^\circ$ ruota): $68.19\%$
- **ET-17 (2025)**:
  - Polinomio: $\% \text{Ack}(\delta_{sw}) = 1.4137 \times 10^{-3} \cdot \delta_{sw}^2 + 64.302\%$
  - $a_0 = 64.30\%$ (64.3% nominale a riposo)
  - A fondo corsa ($\pm 125^\circ$ volante / $\pm 30.0^\circ$ ruota): $86.41\%$

### 23.3 Implementazione nei Solutori Dinamici 14-DOF e Real-Time
Nei modelli fisici della suite (`vehicle_14dof.py`, `interactive_driver_14dof.py`, `params_vehicle.py`), il software non calcola **mai** la divisione per zero $\frac{\Delta\delta_{\text{reale}}}{\Delta\delta_{\text{ideale}}}$.
Invece, gli angoli ruota vengono calcolati direttamente interpolando tramite il fattore continuo e regolare $\text{ack\_pct}(\delta) \in [0, 1]$:
$$\delta_{\text{ack,fl}} = \arctan\!\left(\frac{L \tan\delta}{L - \tfrac{t_f}{2} \tan\delta}\right), \quad
\delta_{\text{ack,fr}} = \arctan\!\left(\frac{L \tan\delta}{L + \tfrac{t_f}{2} \tan\delta}\right)$$
$$\delta_{fl} = \delta + \text{ack\_pct}(\delta) \cdot (\delta_{\text{ack,fl}} - \delta), \quad
\delta_{fr} = \delta + \text{ack\_pct}(\delta) \cdot (\delta_{\text{ack,fr}} - \delta)$$
A $\delta \to 0$, sia $\delta$ che $(\delta_{\text{ack}} - \delta)$ tendono a $0$, garantendo $\delta_{fl} = \delta_{fr} = 0$ con continuità perfetta, zero singolarità numeriche e assoluta stabilità ad ogni frequenza (fino a 500 Hz nel simulatore).

### 23.4 Visualizzazione nella GUI & Tabella Sinottica 16 Curve
- **MDO Target Suite (`main_gui_classic.py`)**:
  - `tab_kinematics` include il selettore di layout:
    - *8 Grafici (Completo con Ackermann & Sterzo)*: Subplot 7 mostra la curva regolarizzata $\% \text{Ack}(\delta_{sw})$ con evidenziazione didattica della zona asintotica Adams ($|\delta| < 8^\circ$); Subplot 8 mostra gli angoli ruota effettivi $\delta_{fl}, \delta_{fr}$ e la differenza $\Delta\delta$.
    - *6 Grafici (Heave & Roll)*: Solo sospensioni verticali e rollio.
    - *2 Grafici (Dettaglio Sterzo)*: Focus dedicato su Ackermann e angoli ruota sterzanti.
- **Inspector Parametri Attivi (`vehicle_parameters_inspector_dialog.py`)**:
  - Scheda 1 aggiornata a **16 Curve Cinematiche (Adams Car)**: la riga 16 riporta la curva di Ackermann, la formula polinomiale, la pendenza parabolica $a_2$ e il valore nominale $a_0$.
- **GUI Professionale (`main_gui_pro.py`)**:
  - Pulsante rapido `"Parametri Attivi"` nell'header tecnico e voci dedicate nel menu *Strumenti*.

---

## 24. Rapporto di Sterzo & Conversione Angolo Volante–Ruota (`params_vehicle.py`)

Il sistema di sterzo è parametrizzato da:
- `steering_ratio`: Rapporto volante/ruota $[-]$ (es. $\pm 135° / \pm 27° = 5.0$)
- `steering_wheel_max_deg`: Escursione massima volante $[°]$

La proprietà derivata `road_wheel_max_deg` calcola l'angolo massimo alla ruota:

$$\delta_{\text{road,max}} = \frac{\delta_{\text{sw,max}}}{\text{steering\_ratio}}$$

Questo valore è usato automaticamente come limite nelle curve di Handling (`generate_handling_sweep`), nelle Mappe DNA e nei diagrammi MMM. Le Mappe DNA mostrano sia l'angolo alla ruota (asse primario) sia l'angolo al volante (annotazione secondaria) per una lettura immediata del pilota.

---

## 25. Accoppiamento Rollio-Imbardata $I_{xz}$ (`vehicle_14dof.py`)

Il modello a 14-DOF supporta il **prodotto d'inerzia incrociato** $I_{xz}$ che accoppia le dinamiche di rollio ($\phi$) e imbardata ($r$):

$$I_x \ddot\phi - I_{xz} \dot r = M_x, \quad I_z \dot r - I_{xz} \ddot\phi = M_z$$

La soluzione simultanea è ottenuta invertendo il sistema:

$$\ddot\phi = \frac{I_z M_x + I_{xz} M_z}{I_x I_z - I_{xz}^2}, \quad
\dot r = \frac{I_x M_z + I_{xz} M_x}{I_x I_z - I_{xz}^2}$$

**Note**: Per vetture FSAE con distribuzione massa simmetrica, $I_{xz} \approx 0$ (standard di dinamica del veicolo). Con `Ixz = 0.0` (default), le equazioni si riducono esattamente alla forma disaccoppiata $\ddot\phi = M_x/I_x$, $\dot r = M_z/I_z$, garantendo zero regressione sulle simulazioni esistenti.

---

## 26. Modulo di Propagazione dell'Incertezza (`mdo_uncertainty.py`)

### 26.1 Sweep di Handling con Warm-Start
La funzione `run_handling_sweep` utilizza un problema OpenMDAO con `SteadyStateGroup` per risolvere l'equilibrio a 3-DOF in sequenza su una rampa di angoli di sterzo:

1. Ad ogni passo $\delta_i$, il solutore Newton viene inizializzato con $r_{i-1}$ e $\beta_{i-1}$ (warm-start)
2. Quando $\delta \approx 0$, il guess iniziale è derivato cinematicamente: $r_0 = v_x \delta / L$
3. **Rilevamento spin-out**: Se $|\beta| > 0.5\,\text{rad}$ oppure $|r| > 5.0\,\text{rad/s}$, la curva viene troncata

### 26.2 Inviluppo di Incertezza
La funzione `get_uncertainty_envelope` esegue 3 sweep paralleli:
- Nominale ($\text{LMUY} = 1.0$)
- Caso peggiore ($\text{LMUY} = 1.0 - \text{uncertainty\_pct}/100$)
- Caso migliore ($\text{LMUY} = 1.0 + \text{uncertainty\_pct}/100$)

---

## 27. Modelli di Prove Dinamiche & Simulazione Eventi (`src/mdo/`)

La suite include solutori dedicati per la simulazione e validazione virtuale delle prove dinamiche ufficiali Formula SAE / Formula Student e delle manovre standardizzate ISO/SAE:

### 27.1 Step Steer a Rampa (ISO 7401 / SAE J266) — `mdo_dynamic_events.py`
- **Profilo di Input**: Da sterzo nullo ad ampiezza prefissata $\delta_{\text{target}}$ mediante rampa lineare con tempo di salita $t_{\text{ramp}} \approx 0.08\text{ s}$ (oppure a gradino ideale), mantenuto costante fino a $t_{\text{end}} = 4.0\text{ s}$.
- **Controllore di Velocità**: Controllore PI a compensazione di carico che mantiene la velocità longitudinale $v_x$ costante a fronte della resistenza d'avanzamento $F_{\text{drag}}$ e della spesa energetica di deriva indotta:
  $$T_{\text{drive}} = K_p (v_{\text{target}} - v_x) + K_i \int_0^t (v_{\text{target}} - v_x)\,d\tau$$
- **Analisi Risposta Transitoria**: Estrazione automatica dei tempi di salita $t_{90}$, overshoot d'imbardata $OS\%$, ritardi $\Delta t_{\text{yaw}}$ e $\Delta t_{ay}$, e del fattore di stabilità transitoria $\text{TB-Factor} = |\beta_{ss}| \cdot t_{\text{peak}}$.

### 27.2 Double Lane Change (ISO 3888-1 & ISO 3888-2 / Moose Test) — `mdo_dynamic_events.py`
- **Geometria del Corridoio**:
  1. Corsia d'ingresso: larghezza $1.1 \cdot w_{\text{veh}} + 0.25\text{ m}$, lunghezza $15\text{ m}$.
  2. Corsia di deviazione: spostamento laterale di $3.5\text{ m}$ (o $4.0\text{ m}$ in ISO 3888-1), lunghezza $30\text{ m}$.
  3. Corsia di rientro: larghezza $1.2 \cdot w_{\text{veh}} + 0.25\text{ m}$, lunghezza $25\text{ m}$.
- **Driver Closed-Loop Tracking (Controllore di Traiettoria)**:
  Il pilota virtuale utilizza un predittore lookahead di tipo Stanley o PID laterale accoppiato all'errore d'imbardata:
  $$\delta(t) = \delta_{\text{feedforward}}(s) + K_{p,y} e_y(t + t_{\text{lookahead}}) + K_{p,\psi} (\psi_{\text{ref}} - \psi)$$
  consentendo di verificare la stabilità d'assetto, l'oscillazione residua di coda (*tail wagging*) e il margine anti-ribaltamento (*roll rollover safety*).

### 27.3 Solutore Evento Skidpad Ufficiale FSAE (D 4.1) — `mdo_skidpad.py`
- **Tracciato ad Otto (Figure-8)**:
  Due cerchi concentrici di diametro interno $15.25\text{ m}$ e diametro di mezzeria $D = 18.25\text{ m}$ ($R = 9.125\text{ m}$).
  Sequenza completa regolamentare: Ingresso rettilineo ($15\text{ m}$) $\to$ Cerchio Destro (Giro 1 d'ingresso + **Giro 2 CRONOMETRATO**) $\to$ Cerchio Sinistro (Giro 1 d'ingresso + **Giro 2 CRONOMETRATO**) $\to$ Uscita rettilinea ($25\text{ m}$).
- **Valutazione Stazionaria & Transitoria**:
  - *Quasi-Static Benchmark*: Calcolo della velocità limite di trim $v_{\text{skid}} = \sqrt{a_{y,\max}(R=9.125) \cdot R}$ e del tempo teorico sul giro:
    $$T_{\text{skid,ss}} = \frac{2\pi R}{v_{\text{skid}}}$$
  - *Full Transient 14-DOF/7-DOF*: Integrazione temporale completa con controllore closed-loop PID di sterzatura sul raggio nominale, tenendo conto del transitorio di rollio, della saturazione differenziale degli pneumatici interni/esterni e dell'accumulo termico nei compound.

### 27.4 Solutore Accelerazione 75m (FS-Rules D 3) — `mdo_acceleration.py`
- **Procedura di Partenza & Staging**: Partenza da fermo con ruote motrici allineate, fotocellula a $0.30\text{ m}$ e misurazione del tempo sul traguardo dei $75.0\text{ m}$.
- **Modello di Trazione & Anti-Slip**:
  Integrazione ODE del moto longitudinale con algoritmo di controllo di trazione (TCS) mirato al mantenimento dello scorrimento ottimo $\kappa_{\text{opt}} \approx 0.095$. Il modello ripartisce istante per istante la coppia applicata alle ruote posteriori nel rispetto della curva di coppia/potenza del motore (EMRAX 188HV) e del tetto di 80 kW dell'accumulatore.

### 27.5 Simulatore Banco Dinamico 7-Post Shaker Rig (`src/mdo/mdo_seven_post_rig.py`)
- **Architettura a 7 Attuatori Idraulici**:
  - 4 Attuatori stradali verticali sotto le ruote ($z_{\text{post},fl}, z_{\text{post},fr}, z_{\text{post},rl}, z_{\text{post},rr}$).
  - 3 Attuatori aerodinamici sul telaio per applicare carichi verticali e momenti di rollio/beccheggio aerodinamici ($F_{z,\text{down},f}, F_{z,\text{down},r}, M_{\text{roll,aero}}$).
- **Decomposizione Modale**: Le eccitazioni sinusoidali o chirp ($0.5 \div 35\text{ Hz}$) sono proiettate nei 4 modi canonici del veicolo:
  1. **Heave**: tutti e 4 gli attuatori in fase ($+1, +1, +1, +1$).
  2. **Pitch**: avantreno e retrotreno in controfase ($+1, +1, -1, -1$).
  3. **Roll**: lato sinistro e lato destro in controfase ($+1, -1, +1, -1$).
  4. **Warp**: diagonali incrociate in controfase ($+1, -1, -1, +1$).
- **Metriche di Banco**: Trasmissibilità telaio/ruota (Diagrammi di Bode ampiezza e fase), variazione del carico di contatto a terra $\sigma(F_z)$ (*Contact Patch Load Variation*) per la massimizzazione del grip meccanico, e isolamento dalle irregolarità dell'asfalto.

---

## 28. Trasferimento di Carico Geometrico ed Elastico nelle Mappe DNA (`guiggiani_dna_maps.py`)

Il modello di trasferimento di carico laterale nelle Mappe DNA utilizza la decomposizione completa **geometrica + elastica**, allineata al modello a 14-DOF:

### 28.1 Angolo di Rollio Stazionario
$$\phi_{ss} = \frac{m \cdot a_y \cdot h_{\text{roll}}}{K_{\phi,\text{tot}} - m \cdot g \cdot h_{\text{roll}}}$$

dove $h_{\text{roll}} = h_{cg} - h_{rc,avg}$ e $h_{rc,avg} = (q_1 b + q_2 a) / L$.

### 28.2 Trasferimento Elastico (attraverso molle/barre antirollio)
$$\Delta F_{z,f}^{\text{el}} = \frac{m \cdot a_y \cdot h_{\text{roll}} \cdot \text{LLTD}_{\text{eff}}}{t_f}, \quad
\text{LLTD}_{\text{eff}} = \frac{1}{2}\!\left(\text{LLTD} + \frac{K_{\phi,f}}{K_{\phi,\text{tot}}}\right)$$

### 28.3 Trasferimento Geometrico (diretto attraverso i centri di rollio)
$$\Delta F_{z,f}^{\text{geo}} = \frac{m \cdot a_y \cdot (b/L) \cdot q_1}{t_f}, \quad
\Delta F_{z,r}^{\text{geo}} = \frac{m \cdot a_y \cdot (a/L) \cdot q_2}{t_r}$$

---

## 29. Interfaccia Grafica Professionale (Dynamic Target Suite Pro) & Architettura di Backup

A partire dalla versione 2.0, la suite introduce una completa riprogettazione dell'interfaccia grafica orientata alla massima ergonomia, standardizzazione ingegneristica e personalizzazione per analisi veicolo avanzate.

### 29.1 Architettura dei 5 Workspace Ingegneristici (`main_gui_pro.py`)
In sostituzione dell'annidamento multilivello a schede della versione classica, la nuova interfaccia organizza le 22 viste analitiche in 5 macro-workspace tematici:
1. **Dinamica Quasistatica**: Momento d'imbardata (YMD), overlay multi-velocità con stime di stabilità, inviluppo G-G-V, diagramma di Handling ($a_y/g$ vs $\delta - L/R$), gradiente $K_{us}$ e MAPs (2D/3D e Vehicle DNA).
2. **Dinamica Transitoria & Telemetria**: Skidpad ISO 4138 a raggio costante, prova di accelerazione 75 m, transitorio step steer e manovra di evitamento ostacolo DLC ISO 3888-2, simulazione percorso completo Autocross ed Endurance.
3. **Assetto & Componenti Sospensioni**: Dimensionamento rigidezze ruota $K_w$ e barre antirollio, caratterizzazione banco ammortizzatori Öhlins TTX25 (curva $F-v$ 4-way), dinamica dello sterzo e coppia autoraddrizzante, mappe cinematiche ruota dal digitizer.
4. **Ottimizzazione MDO & Sensibilità**: Sweep parametrico dell'ottimizzatore OpenMDAO SLSQP, analisi di sensibilità 1D e diagramma Tornado dei KPI dinamici, finestra di bilanciamento ottimale e radar a 4 rapporti, sensitività ai centri di rollio.
5. **Dashboard KPI & Punteggi FSAE**: Cruscotto riepilogativo con delta di ottimizzazione, simulatore ufficiale dei punteggi Formula Student 2026, documentazione interattiva del modello fisico.

### 29.2 Comandi di RUN Evidenti ed Esecuzione MDO Unificata
Per garantire massima chiarezza e accessibilità operativa:
- **Top Command Bar (Header)**: collocati in testata e sempre accessibili da qualunque workspace o sub-tab:
  - **`RUN OTTIMIZZAZIONE MDO`** (Pulsante primario `role="primary_run"` ad alto contrasto in blu cobalto/testo bianco, font 10pt grassetto): avvia l'ottimizzazione globale SLSQP/OpenMDAO sui target veicolo.
  - **`VALUTA PARAMETRI`** (Pulsante secondario `role="secondary_run"` ad alto contrasto su fondo bianco con bordo blu): esegue la simulazione immediata e aggiorna tutti i diagrammi sui parametri correnti senza ottimizzazione.
  - Al termine dell'esecuzione, la suite commuta automaticamente sul **Workspace 5 (Dashboard KPI & Punteggi FSAE)** per esporre immediatamente i risultati.
- **Inspector Laterale (Box Esecuzione alla Base)**:
  - Riquadro dedicato **COMANDI DI RISOLUZIONE & RUN** contenente `self.run_btn` e `self.btn_run_constant`, affiancati dal riquadro espandibile della console di output del solutore (`self.log_output`).

### 29.3 Inspector Laterale Parametrico Collassabile (Modalità Full-Canvas)
La barra laterale sinistra racchiude i vincoli dell'ottimizzatore, i comandi per gli eventi dinamici e lo sweep di progetto. Può essere istantaneamente collassata tramite il pulsante in testata o tramite la combinazione di tasti `Ctrl+B`, massimizzando lo spazio utile a disposizione per i canvas grafici.

### 29.4 Motore dei Temi Grafici, Contrasto Elevato e Politica ZERO EMOJI (`ui_theme_engine.py`)
Il motore centralizzato gestisce:
- **Ingegneristico Chiaro (`light` - Predefinito)**: schema ad alto contrasto assoluto per laboratorio, report e calcolo tecnico. Adotta testo nero puro (`#000000`, contrasto WCAG 21:1), ardesia scura (`#0f172a` / `#334155`), bordi netti slate (`#94a3b8`) e grafici Matplotlib coordinati con assi e scritte nere.
- **Motorsport Scuro (`dark`)**: schema a sfondo ardesia (`#0d1117`, `#161b22`) per sessioni notturne di telemetria.
- **Alto Contrasto Monocromatico (`contrast`)**: schema puro bianco su nero per proiettori ed ambienti industriali.
- **Sincronizzazione Matplotlib**: ogni modifica del tema ricalcola lo sfondo di figura e assi, il colore delle etichette, dei tick e delle griglie tecniche (`alpha=0.22`).
- **Densità dell'Interfaccia**: commutabile tra *Standard* (1440p/4K) e *Compatta* (1080p).
- **Politica ZERO EMOJI**: rigorosa eliminazione di ogni glifo unicode informale o decorativo da pulsanti, titoli, schede, label e log, con conversione tecnica standardizzata (`✔` -> `[OK]`, `❌` -> `[ERR]`, `⚠` -> `[!]`).

### 29.5 Ripristino Affidabile dell'Interfaccia Classica (Backup Bidirezionale In-Process)
L'interfaccia originale classica è interamente preservata all'interno del modulo `main_gui_classic.py` ed è collegata in modo affidabile in-process senza dipendere da subprocess o file su disco:
- Dal pulsante **[ Layout Classico (Backup) ]** presente nell'Header e dal menu **Visualizzazione -> Passa a Interfaccia Classica (Backup)**.
- Dall'interfaccia classica, tramite il pulsante **[ Passa a Layout Pro ]** presente nella barra superiore o dal menu **Interfaccia**.
- Da riga di comando: `python main_gui_classic.py` oppure `python main_gui.py --classic`.
- Tramite impostazione persistente in `ui_settings.json` (`preferred_gui`: `"pro"` o `"classic"`).

---

## 30. Valutazione del Bilanciamento Meccanico tramite Rombo di Setting ARB (`mdo_arb_diamond_balance.py`)

Modulo dedicato all'analisi sistematica dell'effetto del **solo bilanciamento meccanico** (distribuzione della rigidezza di rollio tra avantreno e retrotreno e rigidezza a rollio totale) su **performance** e **guidabilità**, utilizzando il modello **14-DOF completo**.

### 30.1 Geometria del Rombo di Setting dell'ARB
Nelle vetture da competizione e Formula Student, la rigidezza di rollio totale per asse è composta dalla quota elastica delle molle e dalla quota regolabile delle barre antirollio (ARB):
$$K_{\phi,f} = K_{\phi,\text{spring},f} + K_{\phi,\text{arb},f}$$
$$K_{\phi,r} = K_{\phi,\text{spring},r} + K_{\phi,\text{arb},r}$$

Il dominio delle regolazioni possibili delle barre ($K_{\phi,\text{arb},f} \in [K_{f,\min}, K_{f,\max}]$, $K_{\phi,\text{arb},r} \in [K_{r,\min}, K_{r,\max}]$), rappresentato nel piano:
- **Asse X**: Rigidezza a rollio totale $K_{\phi,\text{tot}} = K_{\phi,f} + K_{\phi,r}$ [$\text{Nm/rad}$ o $\text{Nm/deg}$]
- **Asse Y**: Ripartizione di rollio anteriore $\text{Ratio}_f = \frac{K_{\phi,f}}{K_{\phi,\text{tot}}}$ [$-$ o $\%$]

forma un quadrilatero convesso caratteristico denominato **Rombo di Setting dell'ARB** con 4 vertici fisici:
1. **$V_1$ (Soft-Soft)**: entrambe le barre al minimo ($K_{\phi,\text{tot}}$ minimo, bilanciamento intermedio).
2. **$V_2$ (Stiff Front / Soft Rear)**: massima rigidezza ARB anteriore e minima posteriore ($K_{\phi,\text{tot}}$ medio-alto, **massimo bilanciamento anteriore**, massimo sottosterzo).
3. **$V_3$ (Stiff-Stiff)**: entrambe le barre al massimo ($K_{\phi,\text{tot}}$ massimo, bilanciamento intermedio).
4. **$V_4$ (Soft Front / Stiff Rear)**: minima rigidezza ARB anteriore e massima posteriore ($K_{\phi,\text{tot}}$ medio-basso, **minimo bilanciamento anteriore**, massimo sovrasterzo).

### 30.2 Metriche di Performance e Guidabilità con Modello 14-DOF
- **Performance**:
  - $a_{y,\max}$ [g]: accelerazione laterale massima sostenibile.
  - $t_{\text{skidpad}}$ [s]: tempo di percorrenza della prova Skidpad Formula Student su raggio ufficiale $R = 9.125\text{ m}$.
  - Gradiente di rollio $\frac{\partial \phi}{\partial a_y}$ [$\text{deg/g}$].
  - Efficienza di aderenza complessiva $\eta_\mu$ [$\%$].
- **Guidabilità (Handling & Drivability)**:
  - Gradiente di sottosterzo standard secondo convenzione SAE J670e / ISO 4138:
    $$K_{us} = \frac{\partial \delta}{\partial a_y} - \frac{L g}{v_x^2} \quad [\text{deg/g}]$$
    con la relazione fondamentale $\rho_y = -K/L$.
  - Reattività al comando di sterzo: tempo di risposta $t_{90}$ [s].
  - Stabilità transitoria: sovraelongazione della velocità d'imbardata (overshoot $\%$) e tempo di assestamento $t_{\text{settle}}$ [s] su manovra Step Steer ISO 7401 a 14 gradi di libertà.
  - Differenziale di utilizzazione attrito: $\Delta \text{Util} = \text{Util}_F - \text{Util}_R$ [$\%$] (garantisce che l'avantreno saturi prima del retrotreno, evitando perdite di controllo improvvise).

### 30.3 Funzionalità di Mappatura e Ottimizzazione
- **Analisi Vertici & Setup Attuale**: valutazione rapida con radar chart comparativo e telemetria transitoria comparata.
- **Mappa 2D Rombo (Contour)**: campionamento a griglia 2D all'interno del poligono con maschera point-in-polygon (`matplotlib.path.Path.contains_point`), generazione di isolinee di aderenza e sottosterzo, e identificazione automatica dello **Sweet Spot** di ottimo compromesso.
- **Sweep 1D Bilanciamento**: variazione progressiva della ripartizione anteriore a rigidezza totale costante.
- **Applicazione Istantanea**: possibilità di applicare con un click i valori ottimali calcolati ($K_{\phi,f}, K_{\phi,r}, \text{lltd}$) direttamente ai parametri attivi della vettura.

### 30.4 Architettura Asincrona & Gestione Configurazioni con Nome
- **Worker Thread Asincrono (`ARBDiamondWorker`)**:
  Tutte le valutazioni 14-DOF (vertici, griglia 2D, sweep 1D) vengono eseguite all'interno di un thread separato `QtCore.QThread`. La GUI principale non si blocca mai, rimanendo reattiva e fluida.
- **Feedback Live & Annullamento**:
  Il worker emette segnali `progress(done, total, msg)` collegati ad una barra di avanzamento (`QProgressBar`) e ad un banner di stato dinamico, con supporto all'interruzione anticipata immediata tramite pulsante **"✖ Annulla"**.
- **Accelerazione Analitica 14-DOF**:
  Per il campionamento 2D denso e lo sweep 1D, il calcolo sfrutta la calibrazione di riferimento iniziale al centro e il solutore stazionario 14-DOF accoppiato alla sensitività fisica di trasferimento di carico laterale, riducendo il tempo di scansione della griglia 9x9 da oltre 140 secondi a soli ~4.5 secondi. Il transitorio numerico RK4 completo viene riservato allo Sweet Spot ottimale e ai vertici caratteristici.
- **Salvataggio Configurazioni con Nome ("💾 Salva con Nome...")**:
  La configurazione dei 4 vertici del rombo e il loro dimensionamento fisico possono essere salvati con nome in `setups/arb_diamond/<nome>.json`. Il dropdown dei preset indicizza dinamicamente i file presenti nella cartella con ricaricamento, salvataggio rapido ed eliminazione gestiti.

---

## 31. Simulatore 14-DOF: Profilo Stradale 3D, Termica Gomme & Telemetria Avanzata (`road_profile.py`, `interactive_driver_14dof.py`)

### 31.1 Architettura Profilo Stradale 3D a Due Livelli (`road_profile.py`)
Il modulo genera e proietta le asperità e la geometria stradale tridimensionale sul modello veicolo a 14 gradi di libertà e sul rendering OpenGL 3D:
- **Livello Macroscopico (Altimetria & Banking)**:
  - Elevazione longitudinale $z(s)$ [m] con profilo armonico continuo.
  - Pendenza longitudinale $grade = \tan(\theta_{road})$ derivata spazialmente da $dz/ds$.
  - Inclinazione laterale (Banking) $\phi_{bank}(s)$ proporzionale alla curvatura $\kappa(s)$ del tracciato o con variazione naturale collinare.
  - Generazione ribbon 3D con quote cordoli e asfalto:
    $$z_{left}(s) = z(s) + \frac{W_{track}}{2} \sin(\phi_{bank}(s))$$
    $$z_{right}(s) = z(s) - \frac{W_{track}}{2} \sin(\phi_{bank}(s))$$
- **Livello Microscopico (Rugosità e Micro-Ondulazioni)**:
  - Generatore multiarminico ad alta frequenza spaziale ($\lambda \approx 1.5 - 4.0\text{ m}$) con ampiezza regolabile ($0 - 25\text{ mm}$).
  - **Decorrelazione laterale**: sfasamento d'onda tra traccia sinistra e traccia destra che eccita asimmetricamente i triangoli delle sospensioni (travel $z_{fl}, z_{fr}, z_{rl}, z_{rr}$), il rollio, il beccheggio e le variazioni dinamiche di carico normale $F_z$.
- **Opzionalità e Controlli Indipendenti**:
  Entrambi i livelli sono attivabili/disattivabili in modo totalmente indipendente tramite checkbox dedicati ("Altimetria & Banking 3D (Macro)" e "Micro-Ondulazioni & Rugosità (Micro)"), corredati da slider per ampiezza dislivelli, banking e rugosità.

### 31.2 Integrazione Termica Disaccoppiata & Temperatura di Partenza Gomme
- **Frequenza Fisica RK4 Selezionabile fino a 100 Hz**:
  Possibilità di ridurre il passo d'integrazione del modello 14-DOF a 100, 150, 200, 250, 333, 500 Hz per alleggerire il carico CPU su macchine meno performanti.
- **Frequenza di Integrazione Termica Separata**:
  Le dinamiche termiche dei pneumatici hanno costanti di tempo nell'ordine dei secondi (inerzia termica del battistrada). L'integrazione del modello termico (bulk 1-zona o 3-strati superficiale/battistrada/carcassa) è disaccoppiata con frequenza selezionabile tra 10, 25, 50, 100 Hz oppure sincrona con il ciclo RK4 del veicolo.
- **Temperatura di Partenza Gomme $T_{start}$**:
  Impostabile liberamente via SpinBox (10°C - 95°C) o preset rapidi (25°C Fredda, 45°C Calda, 56.2°C Ottimale Termocoperte TTC). All'avvio o reset del simulatore, tutti gli strati termici vengono inizializzati a $T_{start}$.

### 31.4 Editor Interattivo del Profilo Stradale 3D (`RoadProfileEditorDialog`)
- **Immissione ed Editing Quote & Banking da Utente**:
  Integrato il dialog dedicato `RoadProfileEditorDialog`, accessibile sia dalla barra laterale del simulatore ("✏ Modifica Profilo 3D (Quote & Banking)...") sia dal disegnatore spline ("Profilo 3D (Quote & Banking)..."):
  - Permette di definire stazioni altimetriche arbitrarie composte da progressiva curvilinea $s$ [m], quota altimetrica $Z$ [m] e inclinazione laterale di banking $\phi_{bank}$ [°].
  - Ricostruzione del profilo continuo tramite spline cubica $C^2$ con derivata prima spaziale esatta per la pendenza longitudinale:
    $$grade(s) = \frac{dz}{ds} = \tan(\theta_{road}(s))$$
  - Condizioni al contorno periodiche (`bc_type='periodic'`) per circuiti chiusi ad anello, garantendo continuità perfetta di quota, pendenza e curvatura sulla linea di partenza/arrivo.
- **Visualizzazione Grafica Live & Preset Territoriali**:
  - Canvas Matplotlib integrato a doppio asse con aggiornamento istantaneo alla modifica dei valori in tabella:
    * Grafico superiore: profilo quota $Z(s)$ con ombreggiatura e marcatori numerati per ogni stazione.
    * Grafico inferiore: profilo di banking $\phi_{bank}(s)$ con linea di riferimento a 0°.
  - Preset geografici integrati: *"Piatto (2D)"*, *"Collinare (Onde & Banking)"*, *"Circuito di Montagna (Salita/Discesa ±15m)"*, *"Curve Sopraelevate (Banked Corners 10°)"*, *"Rollercoaster (Dossi & Avvallamenti)"*.
  - Funzionalità di generazione stazioni equispaziate, azzeramento a profilo piatto, ordinamento automatico per coordinata $s$, importazione ed esportazione persistente in formato CSV e JSON.

### 31.5 Rifinitura Force Feedback & Sensazione della Strada (Anti-Jerk & Risposta Tattile)
- **Eliminazione degli Scatti Meccanici (Anti-Jerk Continuous Smoothing)**:
  - Rimossa la discontinuità a gradino del 75% nel calcolo delle vibrazioni a favore di un filtro passa-basso esponenziale continuo asimmetrico:
    $$\text{Attacco rapido: } \bar{u}_{k} = \bar{u}_{k-1} + 0.35 (u_{target} - \bar{u}_{k-1}) \quad (\tau \approx 12\text{ ms})$$
    $$\text{Rilascio progressivo: } \bar{u}_{k} = \bar{u}_{k-1} + 0.16 (u_{target} - \bar{u}_{k-1}) \quad (\tau \approx 30\text{ ms})$$
  - Rate limiting hardware USB tarato su ~70 Hz (intervallo minimo 14 ms tra chiamate Win32/XInputSetState), con trasmissione immediata al superamento di una variazione significativa ($\Delta u > 2500$ su scala 65535). Elimina la saturazione della coda interrupt USB di Windows e il cogging meccanico dei rotori.
- **Comunicazione Tattile della Strada & Perdita di Aderenza**:
  - **Sensazione Fondo Stradale (Light Motor)**: Combina il microrilievo dei punti di contatto delle 4 ruote ($wh\_heights$), l'effetto di pendenza dinamica $|grade \cdot v_x|$ e la tessitura d'attrito dell'asfalto proporzionale alla velocità.
  - **Perdita di Aderenza Anteriore / Sottosterzo (Light Motor)**: Innesco progressivo basato sull'eccedenza di slittamento rispetto all'angolo di picco ($\alpha_{peak} = 6.5^\circ$) e saturazione di aderenza ($util > 78\%$), modulato da portante armonica continua a 32 Hz ($0.70 + 0.30\sin(\omega t)$).
  - **Perdita di Aderenza Posteriore / Sovrasterzo (Heavy Motor)**: Innesco immediato su angolo di assetto cassa $|\beta| > 1.5^\circ$, saturazione ruote posteriori ($util_r > 78\%$) e gradiente d'imbardata, azionando il motore pesante con portante a 22 Hz per avvisare istantaneamente il pilota di controsterzare.
  - Aggiunti controlli di guadagno in GUI: slider *"Feedback di Fondo / Texture Asfalto"* (default 35%) e *"Sensibilità Perdita Aderenza"* (0 - 200%, default 100%).

### 31.6 Risoluzione Bug Autosterzatura a Sinistra (Simmetria Angoli di Convergenza / Toe)
- **Diagnosi del Problema sull'Assetto `et18_MSFront`**:
  L'assetto `et18_MSFront.json` impiega valori statici di convergenza asimmetrici per asse (`toe_f_static = +0.5°` toe-in, `toe_r_static = -1.0°` toe-out).
  Nelle formulazioni degli angoli di deriva di `vehicle_14dof.py`, `vehicle_7dof.py` e `interactive_driver_14dof.py`, la convergenza della ruota destra veniva sommata con segno concorde a quella della ruota sinistra:
  $$\alpha_{fr} = \dots + toe_{fr} \quad \text{(ERRATO)}$$
  Questo provocava una sterzatura a granchio (entrambe le ruote puntate a sinistra), generando una coppia d'imbardata spontanea di oltre $1100\text{ Nm}$ a ruote dritte ($r_{dot} = 10.86\text{ rad/s}^2$).
- **Correzione Cinematica**:
  Corretta la convenzione di segno per le ruote del lato destro:
  $$\alpha_{fr} = \dots - toe_{fr}$$
  $$\alpha_{rr} = \dots - toe_{rr}$$
  A $15\text{ m/s}$ in rettilineo a volante centrato, l'accelerazione angolare d'imbardata risulta rigorosamente $r_{dot} = 0.00\text{ rad/s}^2$, garantendo che la vettura proceda in perfetto rettilineo senza derive anomale.

### 31.7 Disaccoppiamento Rigoroso e Layout UI Frequenze (14-DOF RK4 Physics & Tire Thermal)
- **Separazione Visiva e Funzionale**:
  Nella barra laterale di controllo del simulatore è presente il gruppo dedicato `Frequenze di Calcolo (Fisica 14-DOF & Termica)` che separa nettamente i due controlli:
  - `Frequenza Fisica ODE 14-DOF (RK4)`: 100 Hz, 150 Hz, 200 Hz, 250 Hz, 333 Hz, 500 Hz (default 500 Hz).
  - `Frequenza Degrado Termico Gomme`: Sincrona (Fisica RK4), 10 Hz, 25 Hz, 50 Hz, 100 Hz.
  - Indicatore di stato live: etichetta dinamica che riporta il carico computazionale e la frequenza effettiva.
- **Risoluzione Sovrapposizioni UI**:
  Eliminata ogni collisione di griglia layout tra i selettori; i due menu a discesa operano in modo completamente indipendente.

### 31.8 Sistema Preset Force Feedback (FFB) & Calibrazione Softened Anti-Rattling
- **Preset FFB Preconfigurati & Gestione Salvataggio/Caricamento JSON**:
  - `Bilanciato (Consigliato)`: feedback armonico progressivo con texture fondo 20%, perdita aderenza 75%, deadzone 4%.
  - `Morbido / Confortevole`: anti-vibrazioni attenuato per guida fluida, texture 10%, perdita aderenza 50%.
  - `Reattivo / Sportivo`: alta reattività per volanti direct drive e tracciati veloci, texture 35%, perdita aderenza 100%.
  - `Puro Grip (Solo Limite Gomme)`: zero vibrazioni stradali, focus esclusivo sulla saturazione degli pneumatici.
  - `Disattivato`: disabilita rumble e vibrazioni mantenendo la guida attiva.
  - Pulsanti rapidi `Salva Preset...` e `Carica Preset...` per esportare e importare configurazioni custom in JSON.
- **Addolcimento Dinamico & Anti-Scatto**:
  - Smorzamento del filtro passa-basso asimmetrico: attacco $\alpha=0.20$ ($\tau \approx 25\text{ ms}$) e decadimento $\alpha=0.15$ ($\tau \approx 40\text{ ms}$).
  - Innalzamento delle soglie d'innesco degli impulsi tattili: scrub sottosterzo attivo solo con utilizzo aderenza $> 84\%$ o $|\alpha| > 7.5^\circ$; sovrasterzo snap attivo solo per $|\beta| > 2.5^\circ$.
  - Scaling delle ampiezze ridotto di oltre il 50%, eliminando sbattimenti e clatter violenti sui controller.

### 31.9 Correlazione Bidirezionale Profilo 3D - Circuito
- **Integrazione Geometria del Tracciato in `RoadProfileEditorDialog`**:
  - Il dialog riceve e visualizza la geometria del circuito selezionato suddivisa in 3 subplot coordinati:
    1. `Mappa 2D del Tracciato`: pianta del tracciato con evidenziati i marcatori numerati (#1, #2...) corrispondenti alle stazioni altimetriche.
    2. `Profilo Quota Z(s)`: elevazione longitudinale continua con gradiente di pendenza.
    3. `Profilo Banking & Curvatura |κ(s)|`: banking laterale $\phi_{bank}(s)$ a confronto diretto con la curvatura reale $|\kappa(s)|$.
- **Auto-Banking Intelligente da Curvatura**:
  - Pulsante `⚡ Auto-Banking da Curvatura`: calcola istantaneamente il banking ideale su ciascuna stazione proporzionalmente alla curvatura $\kappa(s)$ della curva:
    $$\phi_{bank}(s) = \text{clip}\left(\kappa(s) \cdot \frac{8.5}{\max|\kappa|}, -12.0^\circ, +12.0^\circ\right)$$
- **Adattamento Automatico Cambio Circuito**:
  - Al cambio di circuito nel selettore principale, le progressive delle stazioni vengono scalate proporzionalmente alla nuova lunghezza $L_{new} / L_{old}$, preservando la continuità della pista.

### 31.10 Badge Prestazioni & Diagnostica Real-Time in Fullscreen (F11 HUD)
- **Visualizzazione Diagnostica su Vista 3D**:
  - In modalità Schermo Intero (`F11`), l'HUD proietta un badge diagnostico semi-trasparente in alto a sinistra sotto il timer:
    * `Fisica RK4`: Frequenza effettiva Hz e tempo di calcolo per step $\Delta t_{step}$ [ms].
    * `Termica Gomme`: Frequenza del solutore termico e modalità (Sync / Hz).
    * `CPU Budget & Lag Indicator`: percentuale di budget temporale consumata rispetto all'intervallo target, con badge cromatico:
      - Verde (`● TEMPO REALE OTTIMALE`): budget $< 70\%$, zero rallentamento.
      - Giallo (`▲ CARICO ELEVATO`): budget tra $70\%$ e $100\%$.
      - Rosso lampeggiante (`⚠️ RALLENTAMENTO / LAG`): budget $> 100\%$, simulazione in ritardo sul tempo reale.

### 31.11 Telemetria Avanzata Giro vs Giro & Asse Distanza vs Tempo (`FullTelemetryWindow`)
- **Selettore Asse Orizzontale X**:
  - `Distanza Pista s [m]`: visualizza i dati in funzione della coordinata curvilinea lungo il nastro d'asfalto, fondamentale per l'ingegneria di pista motorsport.
  - `Tempo t [s]`: cronologia continua convenzionale nel dominio del tempo.
  - `Distanza Totale [m]`: distanza odometrica progressiva cumulata.
- **Modalità di Analisi Giri**:
  - `Tutti i Giri (Sequenziale)`: visione complessiva continua della sessione.
  - `Confronto Giro vs Giro (Overlay)`: sovrappone i singoli giri completati sullo stesso intervallo di ascissa $s \in [0, L_{track}]$ (o $\Delta t \in [0, T_{lap}]$). Ciascun giro è tracciato con colore distinto, con evidenziazione in oro e spessore maggiorato per il **Best Lap**.
  - `Solo Best Lap`: isola esclusivamente il miglior giro della sessione per analisi di dettaglio.
  - `Solo Ultimo Giro`: visualizza l'ultimo giro concluso per verifiche immediate post-run.
- **Copertura Completa dei Canali Ingegneristici e Preset Forze 4 Ruote**:
  Oltre ai diagrammi classici di assetto, la finestra dispone del **Preset 2 ("Forze Ruote 4 Angoli")** che sovrappone su grafici dedicati le singole componenti per ciascuna delle 4 ruote ($F_y$ FL, FR, RL, RR; $F_x$; $F_z$; coppie motrici/frenanti), consentendo di analizzare direttamente la ripartizione e il bilanciamento dinamico di carico e aderenza senza limitarsi al solo modulo risultante $|F|$.

### 31.12 Caricamento Telemetrie (.csv / .json) da Interfaccia Utente
- **Importazione Diretta**:
  La telemetria salvata da qualsiasi sessione precedente può essere ricaricata direttamente tramite il pulsante `Carica Telemetria (.csv / .json)` situato sia nella barra comandi del simulatore 14-DOF sia nella toolbar superiore di `FullTelemetryWindow`.
- **Ricostruzione Automatica delle Informazioni di Sessione**:
  All'importazione del file, il parser:
  - Valida e converte le colonne numeriche (con gestione automatica delle intestazioni e tipi di dato).
  - Ricalcola la partizione dei giri (lap detection) identificando i passaggi sulla linea di traguardo.
  - Ricostruisce le statistiche comparative di sessione e identifica automaticamente il giro record (**Best Lap**).
  - Popola la legenda interattiva e abilita la navigazione rapida tra visualizzazione continua o sovrapposta.

### 31.14 Superficie Stradale 2D (ISO 8608 Classe A), C-DLL Road Contact & Geometria Terreno 3D (`track_manager.py`)
- **Superficie Stradale 2D Dipendente da (s, d)**:
  Abbandonata l'approssimazione a binari 1D fissi: la quota altimetrica varia continuamente muovendosi trasversalmente sulla carreggiata mediante la matrice $Z_{\text{surf}}(s, d)$ sintetizzata in `track_manager.py`:
  - **Densità Spettrale di Potenza Longitudinale (ISO 8608 Classe A)**:
    $$G_d(n_s) = G_d(n_0) \cdot \left(\frac{n_s}{n_0}\right)^{-w}$$
    con $G_d(n_0) = 4 \times 10^{-6}\text{ m}^3$, $n_0 = 0.1\text{ cycles/m}$, $w = 2.0$ e deviazione standard calibrata $\sigma \approx 1.41\text{ mm}$.
  - **Filtro di Coerenza Trasversale (Lorentziano)**:
    $$H_d(n_d) = \frac{2 d_{\text{lat}}}{1 + (2\pi n_d d_{\text{lat}})^2} \iff \rho(\Delta d) = \exp\left(-\frac{|\Delta d|}{d_{\text{lat}}}\right)$$
    con $d_{\text{lat}} = 0.20\text{ m}$, garantendo che profili paralleli con separazione laterale $\Delta d > 0.1\text{ m}$ abbiano correlazione decrescente.
  - **Filtro ad Inviluppo Rettangolare Impronta a Terra**:
    Filtro a media mobile spaziale lungo $s$ con finestra $L_{\text{patch}} = 0.12\text{ m}$ per evitare picchi impulsivi su $\dot{z}_{\text{road}}$.
  - **Motore di Campionamento Bilineare a 1000 Hz (`RoadSurface2D`)**:
    Lookup sub-microsecondo ($< 2\ \mu\text{s}$) con wrapping periodico su circuiti chiusi ($s = L \to 0$) e clamping sui bordi carreggiata.
- **Iniezione Parametrica nel Grafo CasADi C-DLL (Nessuna "Fisica Placebo")**:
  - Le 4 quote ruota $[z_{\text{road},fl..rr}]$ (indici 17..20) e velocità $[\dot{z}_{\text{road},fl..rr}]$ (indici 21..24) sono integrate direttamente nel vettore ausiliario $p$ ($N_p = 25$) dell'ODE CasADi compilata in C DLL (`ode_14dof_63d988c94920a084.dll`), firma `'v11_2d_road_contact'`.
  - La deflessione pneumatico $\text{defl}_i = \text{defl}_{0,i} - z_i + z_{\text{road},i}$ e la forza $F_{z,i} = \max(0, K_{tz} \text{defl}_i + C_{tz}(\dot{z}_{\text{road},i} - \dot{z}_i))$ sono calcolate nativamente all'interno del C solver.
- **Estrusione Terreno 3D (Verge Skirt Mesh & Rimozione Piano Rigido Z=0)**:
  - Rimosso il piano piatto orizzontale a $Z=0$ in `interactive_driver_14dof.py`.
  - Generata la mesh continua di banchina (`extrude_verge_geometry`) lungo i versori binormali $\pm\mathbf{b}(s)$ con larghezza $W_{\text{verge}} = 12.0\text{ m}$ e drenaggio $h_{\text{drop}} = 0.4\text{ m}$ colorata in `#2d4a22`.
  - Texture asfalto procedurale 64x64 mappata su GPU con coordinate curvilinee $u = 1.5s, v = 1.5d$.
- **Telemetria HUD 4 Ruote & Rilevamento Lift-Off**:
  - Display compatto delle 4 ruote con Corsa Sospensione, Deflessione Gomma, Quota Suolo Effettiva, e badge `LIFT-OFF` evidenziato in rosso/oro quando $\Delta z_i \le 0$ ($F_{z,i} \le 1.0\text{ N}$).

### 31.13 Modelli 3D Semplificati: FSAE Classica e SUV Geometrico a Volumi Puliti (`vehicle_3d_models.py`)
- **FSAE Racing Car (`FSAECar3DModel`)**:
  Ripristinato il rendering classico, pulito ed essenziale della monoposto da Formula Student:
  - Monoscocca in fibra di carbonio affusolata con pance laterali (sidepods) per radiatori.
  - Casco pilota sfaccettato e cockpit aerodinamico.
  - Ali anteriore e posteriore a profilo pulito e bracci di sospensione a doppi triangoli sovrapposti.
  - Eliminazione di elementi complessi e poligoni superflui, garantendo altissimi framerate stabili (250-500 Hz).
- **B-Segment Road SUV (`SUVCar3DModel`)**:
  Costruito su una geometria volumetrica rigorosa a parallelepipedi solidi puliti ("a rettangoloni"):
  - Scocca inferiore rettangolare con fascioni e battitacco scuri di protezione.
  - Padiglione abitacolo rettangolare (greenhouse) con vetratura scura e montanti solidi.
  - Mascherina frontale rettangolare con fari LED orizzontali e barra luminosa LED continua posteriore.
  - Pneumatici stradali $215/60\text{ R17}$ con cerchi in lega a razze e cinematica di rotolamento e sterzata indipendente.

### 31.14 Calibrazione Frenata e Termica Pneumatici Piattaforma SUV
- **Dimensionamento Coppia Frenante**:
  La coppia frenante massima totale $T_{\text{brake,max}}$ è parametrizzata nella dataclass `VehicleParams` e impostata a $4800\text{ Nm}$ per il profilo SUV (ripartizione $65\%$ ant. / $35\%$ post.), consentendo decelerazioni superiori a $1.15\text{ g}$ con raggio ruota $R_w = 0.315\text{ m}$ e possibilità di frenata al limite e bloccaggio.
- **Inerzia Termica e Raffreddamento Convettivo Pneumatico**:
  Il modello di degradazione termica `TireDegradationModel` adatta i propri parametri termodinamici in funzione del veicolo:
  - Massa termica pneumatico: $C_{\text{th}} = 16000\text{ J/K}$ (contro i $2000\text{ J/K}$ delle slick da corsa), riflettendo la massa battistrada e carcassa di un pneumatico stradale $215/60\text{ R17}$.
  - Coefficiente di dissipazione convettiva: $h_{\text{conv}} = 32.0\text{ W/K}$ (contro gli $8.5\text{ W/K}$ delle piccole ruote da 10/13" scoperte).
  - Efficienza di generazione calore per isteresi/attrito: ridotta da $0.45$ a $0.22$, stabilizzando le temperature operative delle gomme tra $45^\circ\text{C}$ e $65^\circ\text{C}$ ed eliminando il surriscaldamento incontrollato.

---

## 32. Architettura GUI Professionale di Seconda Generazione (`main_gui_pro.py`) & Lifecycle Qt C++

### 32.1 Struttura Modulare a 5 Workspace Ingegneristici
L'interfaccia grafica di seconda generazione riorganizza gli oltre 24 diagrammi e pannelli analitici in 5 Workspace logici dedicati, con Inspector laterale contestuale e collassabile (modalità Full-Canvas):
- **Workspace 0: Dinamica Quasistatica (`ws_quasistatic`)**:
  YMD Singolo (con barra di regolazione parametri $v_x$, slider, toggle Iso-R e ricalcolo integrata nel tab), YMD Multi-Velocità (overlay a più velocità e derivate di stabilità $\partial M_z/\partial \beta$), Inviluppo G-G-V, Handling Diagram (Pacejka), Gradiente $K_{us}$ (SAE) e MAPs prestazionali L17.
- **Workspace 1: Dinamica Transitoria & Telemetria (`ws_dynamic_events`)**:
  Skidpad ISO 4138, Accelerazione 75m, Step Steer & Double Lane Change (ISO 3888-2), Simulazione Percorso Completo AutoX ed Endurance 22km.
- **Workspace 2: Assetto & Componenti Sospensioni (`ws_chassis_suspension`)**:
  Dimensionamento Molle e ARB, Rombo di Setting ARB 14-DOF, Dyno Ammortizzatori TTX25 MkII FSAE, Geometria di Sterzo & Bracci a Terra, Cinematica Digitizer.
- **Workspace 3: Ottimizzazione MDO & Sensibilità (`ws_mdo_sensitivity`)**:
  Sweep Parametrico e Convergenza OpenMDAO, Sensibilità Monovariabile 1D e Tornado Multi-Parametro, Finestra di Bilanciamento & Radar 4-Rapporti Meccanici/Aero, Sensibilità ai Centri di Rollio.
- **Workspace 4: Dashboard KPI & Punteggi FSAE (`ws_kpi_scoring`)**:
  Dashboard KPI Prestazioni e Delta Target, Calcolo Punteggi Regolamento Ufficiale Formula Student 2026, Guida & Basi Fisiche del Veicolo.

### 32.2 Gestione del Ciclo di Vita Qt C++ e Prevenzione `DeferredDelete`
`MainWindowPro` estende per ereditarietà `main_gui_classic.MainWindow`. In fase di costruzione, la classe base istanzia l'albero completo dei widget, canvas, campi numerici e callback di calcolo.
- **Isolamento dell'Ownership C++ (`takeCentralWidget`)**:
  In Qt C++, chiamare `setCentralWidget(new_widget)` quando è già presente un central widget provoca la marcatura immediata del vecchio central widget per `DeferredDelete`. Al processamento degli eventi Qt, tutti i widget figli (tra cui controlli checkbox, tab e bottoni) venivano deallocati lato C++, scatenando `RuntimeError: wrapped C/C++ object of type ... has been deleted` alle successive chiamate di calcolo MDO.
  Chiamando preventivamente:
  ```python
  self._legacy_central_widget = self.takeCentralWidget()
  if self._legacy_central_widget is not None:
      self._legacy_central_widget.hide()
  ```
  l'ownership del vecchio central widget viene distaccata in modo pulito prima dell'assegnazione del nuovo central widget pro, mantenendo vivi e integri in memoria tutti i controlli della classe base.
- **Sincronizzazione Reattiva dei Tab (`_navigate_to_widget`)**:
  L'hook su `self.tabs.setCurrentWidget` intercetta qualsiasi richiesta di navigazione proveniente dalle routine di calcolo (es. al termine di `run_constant_parameters` o `run_optimization`) commutando automaticamente lo stack principale (`pro_workspace_stack`) al workspace di destinazione (0–4) e attivando il rispettivo subtab interno.
- **Protezione Difensiva `_safe_is_checked`**:
  Tutti i controlli su `QCheckBox` (stato di sweep, ottimizzazione robusta, mappe automatiche, vincolo di sottosterzo, iso-raggi YMD) sono incapsulati nella funzione `_safe_is_checked(widget, default=False)`, che valida lo stato dell'oggetto tramite `sip.isdeleted()` e gestisce le eccezioni senza interrompere i calcoli MDO.

### 32.3 Suite di Test & Integrazione CI
La suite include il modulo di verifica automatizzata [`tests/test_mdo_gui_lifecycle.py`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/tests/test_mdo_gui_lifecycle.py) che certifica:
1. L'assenza di distruzione C++ pendente su tutti i widget critici dopo `sendPostedEvents(DeferredDelete)`.
2. Il corretto instradamento dei 5 workspace tramite `tabs.setCurrentWidget`.
3. Il completamento senza errori della valutazione parametri costanti su modelli 3-DOF, 7-DOF e 14-DOF.
4. La commutazione fluida di temi (Motorsport Dark, Light) e densità di layout (Standard, Compatta).

### 32.4 Header Tecnico a Due Righe & Dimensionamento Antitroncamento Controlli
Nelle risoluzioni orizzontali desktop standard e intermedie (1024–1440px), stipare tutti i selettori di setup, modelli, pulsanti primari di esecuzione MDO e preferenze grafiche in una singola riga orizzontale provocava lo schiacciamento critico dei widget (fino a 49–63px di larghezza), troncando il testo dei pulsanti e rendendo incomprensibili comandi vitali.
L'header è stato ingegnerizzato su **due righe funzionali indipendenti**:
- **Riga 1 (Setup & Modelli Fisica)**: Badge di prodotto, selettore profilo setup con pulsanti rapidi (*Salva*, *Salva con Nome...*, *Sfoglia...*), selettore modello dinamico (3-DOF, 7-DOF, 14-DOF) e file pneumatico Pacejka `.tir`.
- **Riga 2 (Esecuzione MDO, Telemetria & Visualizzazione)**: Pulsanti primari ad alto impatto visivo `btn_header_run` (*RUN OTTIMIZZAZIONE MDO*, min-width 220px) e `btn_header_eval` (*VALUTA PARAMETRI*, min-width 165px), indicatore di stato dinamico contestuale (`lbl_header_status`), e box preferenze con selettore tema, densità, toggle rapido dell'Inspector e pulsante di fallback all'interfaccia classica di backup.
- **Barra Workspace Primari**: I 5 pulsanti di navigazione orizzontale sono vincolati a una larghezza minima di 135px e associati a `stretch=1` nel layout, garantendo una suddivisione equa a 5 settori e impedendo che un pulsante collassi a larghezza nulla.

### 32.5 Visibilità Permanente Controlli Grafici Matplotlib (`NavigationToolbar`)
Nei canvas Matplotlib integrati (`NavigationToolbar2QT`), le icone di navigazione (Home, Back, Forward, Pan, Zoom, Subplots, Save) presentavano due problemi di usabilità:
1. **Comportamento Flat (Hover-Only)**: Di default lo stile Qt Fusion applica `autoRaise(True)` sui `QToolButton` inseriti in una `QToolBar`, rendendo i pulsanti trasparenti e privi di cornice fino al passaggio del mouse. Il sistema impone ora esplicitamente `b.setAutoRaise(False)` su tutti i pulsanti delle toolbar, abbinato a regole QSS con sfondi e bordi in rilievo permanenti.
2. **Icone Invisibili nel Tema Chiaro**: Matplotlib determina se ricolorare le icone SVG vettoriali in bianco o nero interrogando la luminosità della palette Qt `toolbar.palette().color(QPalette.Button).value() < 128`. Poiché lo stile applicato impostava sfondi trasparenti generici, Qt restituiva valore 0 (nero), inducendo Matplotlib a disegnare icone bianche (`#ffffff`) su sfondo bianco chiaro, rendendole invisibili.
   Il problema è risolto tramite:
   - La funzione `apply_app_palette(target, theme_name)` in [`ui_theme_engine.py`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/ui_theme_engine.py), che forza la sincronizzazione corretta dei ruoli `Window`, `Base`, `Button`, `Text` e `ButtonText` su `QApplication`, sulla finestra principale e su tutte le toolbar.
   - La chiamata ad `apply_app_palette()` eseguita rigorosamente *dopo* l'applicazione del foglio di stile QSS (per evitare che il parsing del foglio sovrascriva i colori del pulsante con valori neri).
   - In questo modo le icone dei comandi grafici risultano stabilmente nere (`#000000`) nel tema chiaro e bianche (`#ffffff`) nei temi scuro e ad alto contrasto.

---

## 33. Telecamera Cockpit POV a Cinematica Rigida & Sistema Force Feedback Rumble (`interactive_driver_14dof.py`)

### 33.1 Cinematica Rigida a Zero-Lag per Telecamera Cockpit POV Pilota
Nelle visuali in terza persona (Chase Camera), la telecamera utilizza un filtro passa-basso di smoothing esponenziale sulle coordinate ($x, y$) e sull'angolo di imbardata ($\psi$) per simulare l'elasticità di un braccio virtuale ad inseguimento.
Tuttavia, applicare coordinate smoothed a una telecamera interna ad abitacolo (Cockpit POV) causa un ritardo spaziale $\Delta x \approx 0.5\dots 1.5\text{ m}$ in fase di accelerazione longitudinale, portando la telecamera a slittare all'indietro rispetto al veicolo e inquadrando la vettura dall'esterno.
Per garantire un ancoraggio rigido al $100\%$ privo di qualsiasi dislocazione o ritardo:
1. **Lockstep Immediato**:
   In modalità `camera_mode == "COCKPIT"`, le coordinate del punto di vista sono calcolate direttamente dalle variabili di stato istantanee del solutore RK4 a 250 Hz ($x_{car}, y_{car}, z_{road} + 0.20$), senza alcun filtro ritardante.
2. **Matrice di Trasformazione 3D Esatta**:
   L'orientamento del punto di vista e della direzione di mira viene calcolato mediante la matrice di rotazione solida $R = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)$:
   $$P_{\text{eye}} = \begin{bmatrix} x_{\text{car}} \\ y_{\text{car}} \\ z_{\text{car}} \end{bmatrix} + R \begin{bmatrix} x_{\text{loc}} \\ y_{\text{loc}} \\ z_{\text{loc}} \end{bmatrix}$$
   con coordinate occhi pilota nel telaio $x_{\text{loc}} = +0.14\text{ m}$ (immediatamente dietro al volante Formula Student a $+0.22\text{ m}$), $y_{\text{loc}} = 0.0\text{ m}$, $z_{\text{loc}} = +0.40\text{ m}$ (altezza occhi, mirando oltre il display del volante e attraverso l'arco dell'Halo in titanio).
3. **Ottimizzazione del Frustum e Prevenzione Occlusioni**:
   Nel viewport cockpit, il piano di clipping vicino è impostato a $0.03\text{ m}$ (3 cm) e le geometrie retrostanti (roll hoop, cofano motore posteriore, scarico, diffusore e ala posteriore) vengono automaticamente escluse dal rendering GPU, assicurando che la visuale inquadri esclusivamente il cockpit anteriore, il volante rotante, le sospensioni/ruote anteriori sterzanti e la pista.

### 33.2 Architettura Force Feedback Rumble Standard a Due Motori
In ambiente Windows PC, l'API Win32 XInput gestisce nativamente e in modo affidabile i due motori palmari a masse eccentriche asimmetriche:
- **Motore Bassa Frequenza (Heavy / Sinistro)**: assegnato agli eventi ad alta energia del telaio (sovrasterzo brusco, snap di coda, impatto con cordoli e fondo sconnesso fuori pista).
- **Motore Alta Frequenza (Light / Destro)**: assegnato alla texture stradale fine, al micro-scrub del battistrada al limite di aderenza e alla perdita progressiva di trazione.
- **Affidabilità & Test Rapido**: Rimossa la complessa miscelazione con grilletti aptici proprietari (non supportati dal protocollo XInput standard), garantendo chiamate dirette a `XInputSetState` con latch temporale di sicurezza sul pulsante di test `"⚡ Prova Vibrazione Controller (Rumble)"`.

---

## 34. Digitalizzazione e Integrazione Curve Cinematiche Sospensioni 2026 (ET-18)

### 34.1 Estrazione e Catalogazione Curve da Presentazione Adams/Kinematics
Tutti i grafici cinematici della sospensione 2026 (ET-18) sono stati estratti ad alta risoluzione dal file della presentazione Adams `design_suspension/Suspension_ALL.pptx` e organizzati in modo strutturato:
- **Cartelle Tematiche**: `design_suspension/extracted_curves/`
  - `01_Front_Suspension/`: Camber Gain, Bump Steer / Toe, Caster Gain, KPI, Scrub Radius.
  - `02_Rear_Suspension/`: Camber Gain, Bump Steer / Toe, Kingpin, Scrub Radius.
  - `03_Motion_Ratios_ARB/`: Damper Motion Ratio (F/R), ARB Motion Ratio (F/R) vs Wheel Travel.
  - `04_Jacking_Stationary/`: Jacking Forces & Stationary Roll Center Heights vs Ay.
  - `05_Additional_Characteristics/`: Anti-Features (Anti-Dive, Anti-Squat, Anti-Lift).
- **Cartella Piatta**: `design_suspension/extracted_curves/ALL_CURVES_FLAT/` per facilitare il drag-and-drop nel digitalizzatore di curve.
- **Guida Operativa**: `design_suspension/extracted_curves/README_CURVE_DIGITIZER.md` che specifica i range fisici degli assi $(X, Y)$ per ciascun grafico.

### 34.2 Pipeline di Digitalizzazione Automatica e Fitting Polinomiale (3° Grado)
Le curve della vettura **2026 (ET-18)**, identificate dal colore blu tratteggiato (`B > 110, B - R > 35, B - G > 25`), sono state isolate escludendo le maschere delle legende grafiche e calibrate rispetto al bounding box esatto degli assi numerici:
1. **Unità di Misura SI per il Solutore 14-DOF**:
   - Ingresso: Corsa ruota (Wheel Travel Heave) $z$ in **metri** $[m]$ ($+z$ bump/compressione, $-z$ rebound/estensione).
   - Uscita: Angoli in **gradi** $[^\circ]$ (o radianti internamente), quote in **metri** $[m]$, rapporti e percentuali adimensionali $[-]$.
2. **Polinomi di Grado 3**:
   $$y(z) = a_3 z^3 + a_2 z^2 + a_1 z + a_0$$
   Per i gradienti cinematici puri (`camber_gain_f/r`, `toe_gain_f/r`), il termine noto è vincolato a $a_0 = 0.0$ in modo che al valore statico ($z = 0$) la variazione sia nulla e l'assetto coincida esattamente con `camber_static` e `toe_static`.
3. **Esportazioni Singole**:
   Tutti i 15 modelli polinomiali sono salvati in `design_suspension/digitized_2026_curves/*.json`.

### 34.3 Setup Globali `setups/et18_2026_kinematics.json` & `setups/et17_2025_kinematics.json`
I profili di setup includono i parametri cinematici ufficiali Adams delle vetture ET-18 (2026) ed ET-17 (2025):
- **ET-18 (2026)**:
  - **Anteriore**: Centro di rollio statico $h_{rc} = 46.3\text{ mm}$, Molle $K_s = 59.5\text{ N/mm}$, Wheel Rate $K_z = 37.6\text{ N/mm}$, MR ammortizzatore $0.80$, MR barra $0.40$, Caster $3.0^\circ$, KPI $8.2^\circ$, Anti-Dive $30\%$, Camber statico $-1.5^\circ$.
  - **Posteriore**: Centro di rollio statico $h_{rc} = 77.0\text{ mm}$, Molle $K_s = 61.0\text{ N/mm}$, Wheel Rate $K_z = 38.5\text{ N/mm}$, MR ammortizzatore $0.80$, MR barra $0.408$, Anti-Lift $10\%$, Anti-Squat $-14.5\%$, Camber statico $-1.0^\circ$.
- **ET-17 (2025)**:
  - **Anteriore**: Centro di rollio statico $h_{rc} = 43.9\text{ mm}$, Molle $K_s = 54.0\text{ N/mm}$, Wheel Rate $K_z = 34.2\text{ N/mm}$, Camber statico $-1.6^\circ$, Caster $3.2^\circ$, KPI $8.5^\circ$.
  - **Posteriore**: Centro di rollio statico $h_{rc} = 77.2\text{ mm}$, Molle $K_s = 56.0\text{ N/mm}$, Wheel Rate $K_z = 35.8\text{ N/mm}$, Camber statico $-1.2^\circ$.

---

## 35. Formulazione Centri di Rollio in Funzione del Rollio $\phi$ & Ispezione Parametri Attivi

### 35.1 Formulazione Centri di Rollio $h_{rc}(\phi)$ e Migrazione Laterale $y_{rc}(\phi)$
A differenza delle curve di bump (camber, toe, motion ratio) che dipendono dall'escursione ruota in heave $z$, l'altezza verticale dei centri di rollio $h_{rc}$ e la loro migrazione trasversale $y_{rc}$ dipendono primariamente dall'angolo di rollio del telaio $\phi$ [deg], come riportato direttamente dalle simulazioni Adams Car:
1. **Convenzione Unità & Coordinate**:
   - Ingresso: Angolo di rollio telaio $\phi_{\text{deg}} = \phi \cdot \frac{180}{\pi}$ in gradi $[^\circ]$.
   - Uscita: Quota verticale $h_{rc}$ in metri $[m]$ e spostamento trasversale $y_{rc}$ in metri $[m]$ dal piano di mezzeria vettura.
2. **Polinomi di Calibrazione Adams Car (2026 vs 2025)**:
   - **ET-18 (2026)**:
     - $h_{rc,f}(\phi) = 0.0463 - 0.000467 \cdot \phi^2$ ($[0.0, -0.000467, 0.0, 0.0463]$)
     - $h_{rc,r}(\phi) = 0.07693 - 0.000502 \cdot \phi^2$ ($[0.0, -0.000502, 0.0, 0.07693]$)
     - $y_{rc,f}(\phi) = 0.06453 \cdot \phi$ ($[0.0, 0.0, 0.06453, 0.0]$) -> $64.53\text{ mm/deg}$
     - $y_{rc,r}(\phi) = 0.04213 \cdot \phi$ ($[0.0, 0.0, 0.04213, 0.0]$) -> $42.13\text{ mm/deg}$
   - **ET-17 (2025)**:
     - $h_{rc,f}(\phi) = 0.0439 - 0.000600 \cdot \phi^2$ ($[0.0, -0.000600, 0.0, 0.0439]$)
     - $h_{rc,r}(\phi) = 0.07721 - 0.000538 \cdot \phi^2$ ($[0.0, -0.000538, 0.0, 0.07721]$)
     - $y_{rc,f}(\phi) = 0.07367 \cdot \phi$ ($[0.0, 0.0, 0.07367, 0.0]$) -> $73.67\text{ mm/deg}$
     - $y_{rc,r}(\phi) = 0.04113 \cdot \phi$ ($[0.0, 0.0, 0.04113, 0.0]$) -> $41.13\text{ mm/deg}$
3. **Integrazione Fisica**:
   - In `vehicle_14dof.py` (`build_trim_residual`, `compute_equilibrium`, `ode`): $h_{rc}$ e $y_{rc}$ vengono valutati istantaneamente su `phi_deg`.
   - In `mdo_dynamic_events.py` (`simulate_step_steer`, `simulate_double_lane_change`): l'altezza dei centri di rollio è determinata ad ogni passo integrativo da $\phi$.
   - In `interactive_driver_14dof.py`: il thread real-time a 500 Hz calcola $h_{rc,f/r}$ e $y_{rc,f/r}$ ed emette la telemetria live per il display HUD e i grafici dedicati.

### 35.2 Dialogo Unificato di Ispezione Parametri Attivi (`vehicle_parameters_inspector_dialog.py`)
Per consentire a ingegneri e piloti di verificare istantaneamente lo stato esatto di tutti i parametri caricati in memoria sia nella suite MDO che nel simulatore:
- **`ActiveVehicleParametersInspectorDialog`**: Finestra modale moderna con barra di ricerca istantanea, suddivisa in 5 schede:
  1. **📐 15 Curve Cinematiche (Adams Car)**: Tabella sinottica completa con nome curva, variabile indipendente ($z$ [m] o $\phi$ [deg]), valore statico a riposo, 4 coefficienti del polinomio $[a_3, a_2, a_1, a_0]$, pendenza primaria e indicatore visivo di stato (attiva / disattivata).
  2. **🏎️ Assetto, Masse & Geometria**: Passo, carreggiate, massa complessiva, ripartizione $wd_f$, coordinate CG, inerzie telaio e ruote, angoli statici (camber, toe, caster, KPI), scrub radius, braccio a terra e Ackermann.
  3. **⚙️ Sospensioni & Rigidezze**: Rigidezze a rollio $K_\phi$, smorzamento $C_\phi$, wheel rate $K_z$, ripartizione LLTD, molle, tamponi finecorsa (bumpstop) e valving ammortizzatori.
  4. **💨 Aerodinamica & Powertrain**: Superficie $S$, coefficienti $C_z, C_d$, bilancio aero, potenza e coppia motore, rapporto di trasmissione finale, ripartizione frenata, batteria e rigenerazione.
  5. **🔘 Pneumatici (Pacejka MF6.1)**: Parametri completi del file `.tir` attivo e moltiplicatori di aderenza ($LMUX, LMUY$).
- **Funzionalità di Esportazione**:
  - Pulsante `"📋 Copia Tabella (TSV)"` per incollare i dati direttamente in Microsoft Excel o Google Sheets.
  - Pulsante `"💾 Esporta Setup Attivo (JSON)"` per salvare un dump completo e riproducibile dei parametri correnti.

### 35.3 Accesso e Feedback Visivo Permanente nelle GUI
- **MDO Target Suite (`main_gui_classic.py`)**:
  - Pulsante permanente `"📋 Parametri Attivi"` collocato nella barra superiore dei profili.
  - Selettore preset rapido (`ET-18 2026` / `ET-17 2025`), pulsante caricamento pack cartella e pulsante `"📋 Tabella Sinottica 15 Curve"` nel tab Cinematica Sospensioni.
  - Banner permanente con stato di caricamento e diagnostica numerica dei parametri chiave.
  - Subplot dedicati con visualizzazione corretta di $h_{rc}(\phi)$ e $y_{rc}(\phi)$ su $\phi \in [-2^\circ, +2^\circ]$.
- **Simulatore di Guida 14-DOF (`interactive_driver_14dof.py`)**:
  - Scheda di stato `"Cinematica Attiva"` nella tab di calibrazione fisica.
  - Pulsante `"📋 Ispeziona Tutti i Parametri Attivi Simulatore"` per apertura immediata dell'inspector.
  - Indicatore dinamico in tempo reale dei centri di rollio anteriori e posteriori (`lbl_rc_live`) nell'HUD laterale di telemetria.

---

## 36. Architettura di Test, Invarianti Fisici & Benchmark di Performance

La suite adotta una metodologia di collaudo basata sul principio di **falsificazione fisica**: ogni test unitario deve verificare un invariante matematico o un limite analitico noto, anziché limitarsi a verificare che il codice non generi eccezioni non gestite.

### 36.1 Invarianti Fisici e Soglie Analitiche
1. **Roll Gradient Analitico (`tests/test_14dof.py`)**:
   $$\left(\frac{\phi}{a_y}\right)_{\text{analitico}} = \frac{m \cdot h_{\text{arm}}}{K_{\phi,\text{tot}}} = \frac{292 \times (0.270 - 0.0616)}{14000} \cdot \frac{180}{\pi} \cdot 9.81 \approx 2.44\text{ deg/g}$$
   - Bound di regressione ristretto a $[1.9, 3.2]\text{ deg/g}$ ($\pm 30\%$) per coprire migrazione del roll center e carico aero a $15\text{ m/s}$. Risultato calcolato dal solutore: $2.526\text{ deg/g}$.
2. **Sensitività Sottosterzo al Carico Statico ($K_{us}$)**:
   $$K_{us} = \frac{m}{L}\left(\frac{b}{C_{\alpha,f}} - \frac{a}{C_{\alpha,r}}\right)$$
   - Verificata la monotonicità rispetto a $wd_f$: posteriore ($wd_f=0.45, K_{us} = -0.163\text{ deg/g}$), neutro ($wd_f=0.50, K_{us} = +0.005\text{ deg/g}$), anteriore ($wd_f=0.55, K_{us} = +0.173\text{ deg/g}$).
3. **Monotonic Buildup d'Imbardata**:
   - Durante la rampa di sterzo in step-steer 14-DOF ($t \in [0.505, 0.650]\text{ s}$), $\dot{r} \ge 0$ (smorzamento positivo e assenza di instabilità numerica).
4. **Conservazione dell'Energia in Veleggio (Glide)**:
   - A coppia motrice e frenante nulle, la velocità $v_x$ decade monotonicamente a causa del drag aerodinamico ($F_{\text{drag}} = \frac{1}{2}\rho S C_d v_x^2$).
5. **Modello di Pressione MF6.1 (`tests/test_tire_mf61.py`)**:
   - Pressione nominale calibrata ($p_0 = 1.25\text{ bar}$, $dp_i = 0$) come picco di aderenza; test del comportamento al limite per $F_z \to 0$ (assenza di divisioni per zero o propagazione `NaN`).

### 36.2 Benchmark Prestazionali (`tests/test_benchmark_performance.py`)
Tre gate prestazionali con marcatore `@pytest.mark.benchmark`:
- **Valutazione Residui Trim 14-DOF**: $< 5.0\text{ ms}$ (misurato: $\approx 0.015\text{ ms}$).
- **Calcolo Mappe DNA del Veicolo**: $< 250.0\text{ ms}$ (misurato: $\approx 16.5\text{ ms}$).
- **Integrazione Manovra Step-Steer 2.0s**: $< 2000.0\text{ ms}$ (misurato: $\approx 613\text{ ms}$).

### 36.3 Esecuzione dei Test
```bash
# Esecuzione rapida di tutta la suite fisica e di ottimizzazione (128 test)
.\venv\Scripts\python.exe -m pytest -v --ignore=tests/test_gui_systematic.py

# Esecuzione completa dei test di ciclo di vita GUI
.\venv\Scripts\python.exe -m pytest tests/test_mdo_gui_lifecycle.py tests/test_gui_systematic.py -v

# Esecuzione dei soli benchmark prestazionali
.\venv\Scripts\python.exe -m pytest -m benchmark -v
```

---

## 37. Modularizzazione Front-End & Pipeline CI/CD (GitHub Actions)

### 37.1 Architettura Modulare della GUI
Per mantenere alta la manutenibilità e ridurre la dimensione monolitica dei file grafici, i componenti trasversali sono separati in:
- **`gui_canvas.py`**: Incapsulamento del canvas Matplotlib `MplCanvas` con policy di sizing reattiva.
- **`gui_dialogs.py`**: Dialogo `ChartGuideDialog`, `PopoutPlotDialog` e motore di traduzione scientifica `translate_figure_language` (`FIGURE_IT_TO_EN`).
- **`gui_workers.py`**: Thread in background PyQt5 (`SensitivityWorker`, `ARBDiamondWorker`, `CollocationWorker`) per calcoli asincroni non bloccanti.

### 37.2 Pipeline di Continuous Integration (`.github/workflows/ci.yml`)
La pipeline GitHub Actions viene eseguita automaticamente ad ogni push e pull request sui branch `main` e `master`:
1. **Job `physics-and-benchmarks`**: installazione dipendenze ed esecuzione dei 128 test matematici e fisici.
2. **Job `gui-lifecycle`**: verifica del ciclo di vita headless dei widget e delle simulazioni dinamiche.
3. **Job `standalone-build`**: compilazione dell'eseguibile standalone con PyInstaller e verifica di integrità dell'eseguibile `dist/MDO_Target_Suite/MDO_Target_Suite.exe`.

---

---

## 38. Simmetria Full-Lock, Camber-Steer & 19 Curve Cinematiche Adams Car

### 38.1 Risoluzione Asimmetria a Fondo Corsa (Decoupled Caster & KPI Jacking)
Nelle manovre a fondo corsa sterzo ($\pm 135^\circ$ volante, circa $\pm 25^\circ$ ruota), la cinematica dell'avantreno è soggetta a due effetti verticali indipendenti:
1. **Caster Jacking (Modo Differenziale / Rollio)**:
   $$\Delta z_{\text{caster}} = -d_{\text{spindle}} \cdot \sin(\nu_{\text{caster}}) \cdot \sin(\delta)$$
   È una funzione **dispari** di $\delta$: inverte di segno tra sterzata a sinistra e a destra, trasferendo carico verticalmente in diagonale tra ruota interna ed esterna.
2. **KPI Jacking (Modo Comune / Heave)**:
   $$\Delta z_{\text{kpi}} = +r_{\text{scrub}} \cdot \sin(\sigma_{\text{kpi}}) \cdot (1 - \cos(\delta))$$
   È una funzione **pari** di $\delta$: è sempre positiva ($\ge 0$) e solleva contemporaneamente **entrambi** i fusi ruota anteriori rispetto al telaio (heave lift comune del muso).

**Separazione Fisica nel Solutore 14-DOF**:
I carichi a terra vengono calcolati disaccoppiando il modo differenziale dal modo comune:
$$F_{z,fl} = \frac{F_{z,\text{stat},f}}{2} - \Delta F_{z,\text{llt}} + \Delta F_{z,\text{caster}} + \Delta F_{z,\text{kpi}}$$
$$F_{z,fr} = \frac{F_{z,\text{stat},f}}{2} + \Delta F_{z,\text{llt}} - \Delta F_{z,\text{caster}} + \Delta F_{z,\text{kpi}}$$
Questo garantisce la **perfetta simmetria speculare** su tutte le 4 ruote:
$$z_{fl}(+\delta) = z_{fr}(-\delta), \quad F_{z,fl}(+\delta) = F_{z,fr}(-\delta)$$

### 38.2 Modello Camber Steer ($\Delta\gamma_{\text{steer}}$)
La rotazione attorno all'asse del montante inclinato genera una variazione cinematica di campanatura:
$$\Delta\gamma_{\text{steer},fl}(\delta) = P(\delta_{sw,fl}), \quad \Delta\gamma_{\text{steer},fr}(\delta) = P(-\delta_{sw,fr})$$
dove $P(\delta_{sw})$ è il polinomio di 3° grado vincolato a $a_0 = 0.0$ a centro ruota.
- Nel **2025 (ET-17)** con Caster $= 6.0^\circ$, la variazione totale copre $5.3^\circ$ (da $-3.3^\circ$ a $+2.0^\circ$).
- Nel **2026 (ET-18)** con Caster $= 3.0^\circ$, la variazione è dimezzata a $2.4^\circ$ (da $-1.8^\circ$ a $+0.6^\circ$).

### 38.3 Catalogo delle 19 Curve Cinematiche Adams Car
1. `camber_gain_f_coeffs`: $\gamma_f(z)$ vs corsa ruota
2. `camber_gain_r_coeffs`: $\gamma_r(z)$ vs corsa ruota
3. `toe_gain_f_coeffs`: $\delta_{toe,f}(z)$ vs corsa ruota
4. `toe_gain_r_coeffs`: $\delta_{toe,r}(z)$ vs corsa ruota
5. `rollcenter_f_coeffs`: $h_{rc,f}(\phi)$ vs rollio
6. `rollcenter_r_coeffs`: $h_{rc,r}(\phi)$ vs rollio
7. `rollcenter_lat_f_coeffs`: $y_{rc,f}(\phi)$ vs rollio
8. `rollcenter_lat_r_coeffs`: $y_{rc,r}(\phi)$ vs rollio
9. `anti_dive_coeffs`: anti-affondamento frenata
10. `anti_squat_coeffs`: anti-schiacciamento trazione
11. `anti_lift_coeffs`: anti-sollevamento staccata
12. `damper_motion_ratio_f_coeffs`: MR ammortizzatore ant.
13. `damper_motion_ratio_r_coeffs`: MR ammortizzatore post.
14. `arb_motion_ratio_f_coeffs`: MR barra ARB ant.
15. `arb_motion_ratio_r_coeffs`: MR barra ARB post.
16. `ackermann_coeffs`: $\%_{\text{Ack}}(\delta_{sw})$ vs sterzo
17. `caster_trail_f_coeffs`: $t_m(\delta_{sw})$ braccio meccanico vs sterzo
18. `scrub_radius_f_coeffs`: $r_{\text{scrub}}(\delta_{sw})$ raggio a terra vs sterzo
19. `camber_steer_f_coeffs`: $\Delta\gamma_{\text{steer}}(\delta_{sw})$ delta camber vs sterzo
20. `wheel_steer_f_coeffs`: $\delta_{\text{wheel}}(\delta_{sw})$ angolo sterzata ruota ant. vs sterzo volante/pignone (Adams Car)

---

## 39. Modello Transitorio Pneumatico a Lunghezza di Rilassamento (Pacejka Contact Patch Relaxation Length)

### 39.1 Fondamenti Fisici e Formulazione Differenziale
Nei transitori rapidi (ingresso curva, cambi di direzione, transitori di frenata e trazione) la forza sviluppata dall'impronta a terra del pneumatico non reagisce istantaneamente al valore algebrico statico della Magic Formula MF6.1, ma è governata dalla deflessione elastica della carcassa all'impronta di contatto (*Contact Patch Carcass Deflection*).

Il modello a 23 variabili di stato estende l'ODE a 14-DOF con 8 stati di deflessione elastica:
$$[u_{x,fl}, u_{y,fl}, u_{x,fr}, u_{y,fr}, u_{x,rl}, u_{y,rl}, u_{x,rr}, u_{y,rr}]$$

Per ciascuna ruota $i \in \{fl, fr, rl, rr\}$, le velocità di scorrimento reale al mozzo nel frame ruota sono:
$$V_{sx,i} = V_{x,w,i} - \omega_i R_w, \qquad V_{sy,i} = V_{y,w,i}$$

Le equazioni differenziali ordinarie del 1° ordine per le deflessioni longitudinali e laterali sono:
$$\dot{u}_{x,i} = -V_{sx,i} - \frac{|V_{x,w,i}|}{\sigma_x} u_{x,i} = (\omega_i R_w - V_{x,w,i}) - \frac{\sqrt{V_{x,w,i}^2 + \epsilon_v^2}}{\sigma_x} u_{x,i}$$
$$\dot{u}_{y,i} = -V_{sy,i} - \frac{|V_{x,w,i}|}{\sigma_y} u_{y,i} = -V_{y,w,i} - \frac{\sqrt{V_{x,w,i}^2 + \epsilon_v^2}}{\sigma_y} u_{y,i}$$

dove:
- $\sigma_x = 0.15\text{ m}$: lunghezza di rilassamento longitudinale.
- $\sigma_y = 0.22\text{ m}$: lunghezza di rilassamento laterale.
- $\epsilon_v = 10^{-2}\text{ m/s}$: parametro di regolarizzazione liscia CasADi per l'eliminazione delle singolarità a zero.

### 39.2 Comportamento a Veicolo Fermo (Molla Elastica Pura)
A velocità nulla ($V_{x,w,i} \to 0$):
$$\lim_{V_{x,w} \to 0} \dot{u}_{x,i} = -V_{sx,i} = \omega_i R_w - V_{x,w,i}$$
$$\lim_{V_{x,w} \to 0} \dot{u}_{y,i} = -V_{sy,i} = -V_{y,w,i}$$
L'impronta a terra si comporta come una **molla elastica lineare priva di singolarità**:
$$F_{x,i} \approx \frac{C_F}{\sigma_x} u_{x,i} = K_{\text{tire},x} \cdot u_{x,i}$$
$$F_{y,i} \approx \frac{C_\alpha}{\sigma_y} u_{y,i} = K_{\text{tire},y} \cdot u_{y,i}$$
Questo elimina la classica divisione per zero algebrica $\frac{1}{\max(v_x, 0.5)}$, consentendo al simulatore di stabilizzare la vettura da ferma in modo puramente fisico attraverso l'elasticità della gomma.

### 39.3 Formulazione Anti-Windup e Slittamenti Dinamici
Nelle versioni iniziali, applicare una compressione non lineare $\tanh$ direttamente sullo stato $u$ *prima* del calcolo della derivata temporale ($\dot{u}_x = -V_{sx} - \frac{V_{mag}}{\sigma_x} u_{lim}\tanh(u/u_{lim})$) creava un limite superiore rigido al termine di smorzamento pari a $-1.5 V_{mag}$. Durante un testacoda o forte pattinamento con $|V_{sx}| > 1.5 V_{mag}$, la derivata non poteva azzerarsi, portando lo stato integrato $u$ a divergere verso valori abnormi (integrator windup) e mantenendo una trazione fantasma ($F_x \approx 2300\text{ N}$) a farfalla chiusa.

La formulazione corretta calcola la derivata $\dot{u}$ direttamente sullo **stato libero** (garantendo un filtro del 1° ordine linearmente asintoticamente stabile) con una soglia di regolarizzazione a bassa velocità $V_{mag,eff} = \max(V_{mag}, 0.5\text{ m/s})$:
$$\dot{u}_{x,i} = -V_{sx,i} - \frac{V_{mag,eff,i}}{\sigma_x} u_{x,i}$$
$$\dot{u}_{y,i} = -V_{sy,i} - \frac{V_{mag,eff,i}}{\sigma_y} u_{y,i}$$

La saturazione liscia $\tanh$ viene applicata **esclusivamente in uscita** ai rapporti di scorrimento adimensionali passati alla Magic Formula MF6.1 (limite di Coulomb allo strisciamento $\approx 1.5$):
$$\kappa_{dyn,i} = 1.5 \tanh\left(\frac{u_{x,i}}{1.5 \sigma_x}\right), \qquad \alpha_{dyn,i} = \arctan\left(1.5 \tanh\left(\frac{u_{y,i}}{1.5 \sigma_y}\right)\right) \pm \delta_{toe,i}$$
A piccole deflessioni vale esattamente l'asintoto lineare $\kappa \approx u_x / \sigma_x$ e $\tan\alpha \approx u_y / \sigma_y$, mentre a scorrimenti estremi lo stato $u$ non accumula energia fittizia e rientra istantaneamente a zero al cessare dello strisciamento.

### 39.4 Modello di Drag Aerodinamico Vettoriale 2D
Per dissipare correttamente l'energia cinetica in ogni direzione di moto planare (incluso il moto completamente trasversale a $90^\circ$ durante un testacoda), la resistenza aerodinamica opera come vettore 2D planare opposto al vettore velocità del telaio:
$$V_{planar} = \sqrt{v_x^2 + v_y^2 + 10^{-6}}$$
$$F_{drag,x} = \frac{1}{2} \rho S C_d V_{planar} v_x, \qquad F_{drag,y} = \frac{1}{2} \rho S C_d V_{planar} v_y$$
I termini entrano nei bilanci di forza del telaio:
$$\Sigma F_x = \sum_{i} F_{x,i} - F_{drag,x} - F_{rr}, \qquad \Sigma F_y = \sum_{i} F_{y,i} - F_{drag,y}$$

Nel simulatore real-time (`interactive_driver_14dof.py`):
1. **Force Feedback**: La coppia autoallineante $M_z$ (SAT) e il braccio pneumatico $t_p$ sono calcolati da $\alpha_{dyn}$, fornendo una sensazione di sterzo fluida e priva di jitter numerico a bassa velocità.
2. **Haptics Triggers**: I livelli di vibrazione per perdita di trazione (grilletto RT) e bloccaggio in staccata (grilletto LT) utilizzano $\kappa_{dyn}$.
3. **Modello Termico**: Il riscaldamento battistrada sfrutta la velocità di slittamento $V_{\text{slip}}$ calcolata da $\kappa_{dyn}$ e $\alpha_{dyn}$.
4. **Telemetria**: Sono registrati i canali di deflessione $u_{x,i}\text{ [mm]}$ e $u_{y,i}\text{ [mm]}$ per tutte e quattro le ruote (`ux_fl_mm`, `uy_fl_mm`, ecc.).

---

## 40. Cinematica di Sterzo Adams Car (`wheel_steer_vs_pinion_steer`) & Ackermann Effettivo

### 40.1 Vantaggi rispetto alla Percentuale di Ackermann
Nei sistemi di sterzo ad alte prestazioni, la classica percentuale scalare o curva di Ackermann $\%_{\text{Ack}}(\delta_{sw})$ presenta una singolarità asintotica per $\delta_{sw} \to 0$ ($0/0 \to \infty$), richiedendo continue soglie di regolarizzazione. Inoltre, l'angolo pignone/volante non corrisponde a un rapporto fisso lineare con le ruote a terra.

L'estrazione multibody da Adams Car (`01_10_steering_ratio_f__wheel_steer_vs_pinion_steer.jpeg`) fornisce direttamente la curva $C^\infty$ e regolare:
$$\delta_{\text{wheel}} = f_{\text{steer}}(\delta_{sw})$$
dove $\delta_{sw} \in [-125^\circ, +125^\circ]$ è l'angolo di sterzo del pignone/volante e $\delta_{\text{wheel}} \in [-35^\circ, +25^\circ]$ è l'angolo di sterzata della ruota a terra.

### 40.2 Simmetria Speculare Bilaterale
Poiché la curva digitalizzata si riferisce a una singola sospensione (ruota anteriore sinistra FL), la cinematica dell'altra ruota (FR) è data dalla simmetria speculare rispetto all'asse di mezzeria del veicolo:
$$\delta_{FL}(\delta_{sw}) = f_{\text{steer}}(\delta_{sw})$$
$$\delta_{FR}(\delta_{sw}) = -f_{\text{steer}}(-\delta_{sw})$$

Grazie a questa formulazione:
- **Svolta a sinistra ($\delta_{sw} > 0$)**: la ruota interna FL curva con angolo maggiore rispetto alla ruota esterna FR ($\delta_{FL} > \delta_{FR} > 0$). A $\delta_{sw} = +100^\circ$, $\delta_{FL} = +20.49^\circ$ e $\delta_{FR} = +17.60^\circ$, con un Ackermann differenziale di $+2.89^\circ$.
- **Svolta a destra ($\delta_{sw} < 0$)**: la ruota interna FR curva con ampiezza maggiore rispetto alla ruota esterna FL ($|\delta_{FR}| > |\delta_{FL}|$, $\delta_{FR} < \delta_{FL} < 0$). A $\delta_{sw} = -100^\circ$, $\delta_{FR} = -20.49^\circ$ e $\delta_{FL} = -17.60^\circ$, con un Ackermann differenziale di $+2.89^\circ$.
- Entrambi i rendering 3D/2D del simulatore e le equazioni ODE CasADi calcolano e visualizzano l'esatto angolo differenziale, eliminando ogni anomalia grafica o "effetto anti-Ackermann".

---

## 41. Ricalibrazione Modello di Usura Gomme (FSAE Hoosier R20) & Scala Dinamica

### 41.1 Calibrazione del Tasso di Usura
Nel modulo `tire_degradation.py`, il tasso di usura battistrada è modellato sull'energia dissipata per attrito:
$$\text{wear\_rate} = k_{\text{wear}} \cdot s_{\text{wear}} \cdot \frac{P_{\text{slip}}}{F_{z,\text{safe}} \cdot v_{x,\text{safe}}}$$
dove $\frac{P_{\text{slip}}}{F_z \cdot v_x}$ è un rapporto adimensionale tra la potenza di slittamento tangenziale e la potenza del carico normale percorso.

Il coefficiente base $k_{\text{wear}}$ è stato ricalibrato a $k_{\text{wear}} = 2.0 \times 10^{-3}$, garantendo:
- $\sim 3\% - 8\%$ di usura del battistrada dopo 5 km di percorrenza mista con curve ad alto carico laterale.
- $\sim 20\% - 35\%$ di usura in una prova Endurance completa FSAE (22 km), coerente con i rilievi sperimentali di consumo dei pneumatici Hoosier R20 compound.

### 41.2 Moltiplicatore di Accelerazione Usura
Nel simulatore interattivo (`interactive_driver_14dof.py`), l'utente può selezionare la velocità di consumo:
- `1x (Reale Endurance)`: scala standard per simulazioni lunghe.
- `2x (Medio)`: degradazione avvertibile in 10-12 giri.
- `5x (Test Rapido)`: per testare rapidamente il calo di aderenza ($\mu_{\text{eff}} \to \mu_{\text{worn\_min}}$).
- `10x (Sprint Usura)`: per sessioni brevi da 2-3 giri.

---

## 42. Minimappa Tracciato HUD nel Simulatore 14-DOF

### 42.1 Funzionalità & Rendering
L'HUD del simulatore include una minimappa vettoriale integrata ad alte prestazioni:
- **Posizionamento**: Colonna sinistra sotto il Lap Timing (`x = 14, y = 140, w = 210, h = 180`).
- **Ribbon Tracciato**: Geometria del tracciato scalata con aspect ratio preservato e campionamento precalcolato (zero overhead a 60 FPS).
- **Traguardo S/F**: Barra dorata/ambra perpendicolare alla corda di partenza.
- **Marker Veicolo**: Indicatore direzionale a freccia Ferrari Red orientato secondo l'angolo di imbardata reale $\psi$ della vettura, con alone luminoso ciano.
- **Scorciatoia Tastiera**: Tasto rapido `M` per attivare o nascondere la minimappa istantaneamente.

---

## 43. Veicolo Stradale Segmento B (SUV Compatto FWD con Differenziale Aperto) & Modelli 3D Parametrici

### 43.1 Visione & Requisiti Fisici
Accanto alle monoposto ad alte prestazioni di Formula Student (FSAE), la suite integra una categoria veicolo stradale: un **SUV compatto di Segmento B** a **trazione anteriore (FWD)** con **differenziale aperto meccanico**.

Il profilo è calibrato all'interno della Single Source of Truth (`VehicleParams.segmento_b_suv()` in `params_vehicle.py`) e salvato nel file di configurazione predefinito `setups/segmento_b_suv.json`:
- **Massa & Inerzie**: $m = 1250.0\text{ kg}$ (con pilota a bordo e liquidi), ripartizione statica $wd_f = 0.60$ (60% ant / 40% post), $I_z = 1850\text{ kg}\cdot\text{m}^2$, $I_x = 520\text{ kg}\cdot\text{m}^2$, $I_y = 1700\text{ kg}\cdot\text{m}^2$.
- **Geometria**: Passo $L = 2.56\text{ m}$, Carreggiate $t_f = t_r = 1.54\text{ m}$, Altezza centro di gravità $h_{cg} = 0.56\text{ m}$.
- **Pneumatico Stradale Dedicato Pacejka MF6.1 (`Road_215_60R17_Touring.tir`)**:
  - File `.tir` completo standard Pacejka 6.1 (120+ parametri fittati) collocato in `tire_tir/Road_215_60R17_Touring.tir` e referenziato direttamente dal campo `tire_file` di `VehicleParams`.
  - **Carico Nominale Calibrato**: $F_{z0} = \text{FNOMIN} = 3500\text{ N}$ (rispetto ai $1000\text{ N}$ della gomma FSAE). In questo modo il carico di lavoro statico anteriore ($\approx 3680\text{ N}$) e in pieno appoggio ($\approx 6000\text{ N}$) ricade nel range lineare/fisiologico del modello Pacejka ($df_z \in [-0.15, +0.9]$), prevenendo estrapolazioni quadratiche patologiche e saturazioni artificiali.
  - **Dimensioni & Pressione**: Raggio scarico $R_0 = 0.3449\text{ m}$, raggio di rotolamento $R_w = 0.315\text{ m}$, larghezza battistrada $0.215\text{ m}$, pressione nominale $2.3\text{ bar}$ ant / $2.2\text{ bar}$ post.
  - **Aderenza & Rigidezza di Deriva**: Picco laterale $\mu_{y,\text{peak}} \approx 0.98\text{ g}$ a $3.5\text{ kN}$ ($\alpha_{\text{peak}} \approx 6.5^\circ$) e $\mu_{y} \approx 0.94\text{ g}$ a $6\text{ kN}$, rigidezza di deriva d'angolo iniziale $K_{y\alpha} \approx 830\text{ N/deg}$ ($47500\text{ N/rad}$), picco longitudinale $\mu_{x,\text{peak}} \approx 1.05\text{ g}$ a $\kappa \approx 0.09$, e coppia di allineamento $M_z$ di picco pari a $52.2\text{ Nm}$.
  - **Lunghezze di Rilassamento & Moltiplicatori Puliti**: $\sigma_x = 0.28\text{ m}$, $\sigma_y = 0.35\text{ m}$; moltiplicatori nominali `LMUX = 1.0` e `LMUY = 1.0` (lasciando la scala pulita per simulazioni meteo bagnato $0.70$ o usura).
- **Sospensioni Stradali Morbide**: Rigidezze verticali ruota $K_{z,f} = 28000\text{ N/m}$, $K_{z,r} = 26000\text{ N/m}$, rigidezze rollio $K_{\phi,f} = 45000\text{ Nm/rad}$, $K_{\phi,r} = 32000\text{ Nm/rad}$, corsa ammortizzatori $\pm 80\text{ mm}$ con bumpstop a $55\text{ mm}$.
- **Sterzo Stradale**: Rapporto di sterzo $15.5:1$, escursione volante $\pm 500^\circ$ ($\pm 32.25^\circ$ alle ruote a terra).

### 43.2 Drivetrain FWD & Fisica del Differenziale Aperto
Nelle equazioni dinamiche 14-DOF (`vehicle_14dof.py` e `interactive_driver_14dof.py`), la topologia di trazione si adatta al parametro `drivetrain_type`:
1. **Routing Coppia Motrice**: Quando `drivetrain_type == 'fwd'`, la coppia motrice $T_{\text{drive}}$ viene applicata esclusivamente alle ruote anteriori (FL, FR), lasciando le ruote posteriori (RL, RR) puramente trascinate ($T_{\text{drive,RL}} = T_{\text{drive,RR}} = 0$).
2. **Differenziale Aperto**: In linea retta o aderenza simmetrica, la coppia motrice si ripartisce esattamente al 50/50:
   $$T_{\text{FL}} = T_{\text{FR}} = \frac{T_{\text{tot}}}{2}$$
   In curva stretta o su disconnessioni, quando la ruota interna anteriore si scarica ed entra in pattinamento (differenza di scorrimento $\Delta \kappa = |\kappa_{FL} - \kappa_{FR}| > 0.12$), la coppia motrice trasferibile a terra viene fisicamente vincolata dal limite di aderenza della ruota più debole, replicando fedelmente il tipico comportamento dei veicoli a trazione anteriore con differenziale aperto.

### 43.3 Motore Grafico 3D Modulare (`vehicle_3d_models.py`)
La resa grafica 3D del simulatore e dello studio anteprima è stata completamente modularizzata nel componente `vehicle_3d_models.py`:
- **`FSAECar3DModel`**: Monoposto da competizione Formula SAE (musetto a freccia, ali deportanti multi-elemento, cockpit aperto con halo in titanio, cerchi da 10/13 pollici e scarico posteriore).
- **`SUVCar3DModel`**: Carrozzeria volumetrica solida SUV compatto ("a rettangoloni"), skid plates protettivi satinati anteriore e posteriore, mascherina a nido d'ape, proiettori LED con barre DRL, parabrezza inclinato, abitacolo rialzato con barre portatutto sul tetto, portellone hatchback con barra luminosa posteriore a tutta larghezza e ruote stradali maggiorate da 17" con cerchi diamantati a 5 razze.

### 43.4 Studio Anteprima 3D Veicolo (`scripts/preview_car_3d.py`)
Lo studio dedicato 3D offre una piattaforma di visualizzazione e validazione geometrica a 360° con:
- Selettore veicolo al volo: `Formula Student (FSAE)` vs `Piccolo SUV Segmento B (FWD)`.
- Telecamera orbitale con piatto studio adattivo ($r = 3.6\text{ m}$ per SUV vs $2.4\text{ m}$ per FSAE).
- Controllo live escursione sospensioni (fino a $\pm 80\text{ mm}$), angoli di sterzata, HUD utilizzo grip a 4 ruote, modalità wireframe, autorotazione e selezione colore carrozzeria / dettagli.
- Accessibilità istantanea da menu *Strumenti / Tools* in `main_gui_pro.py`, `main_gui_classic.py` e dal pannello setup del simulatore real-time `interactive_driver_14dof.py`.

---

## 44. Powertrain RWD Dual-Motor: Differenziali, Active Torque Vectoring, Controlli TTX25 & 3D Ghost Car

### 44.1 Modello Differenziali RWD & Active Torque Vectoring (TV)
Nel simulatore real-time 14-DOF (`interactive_driver_14dof.py`) e nei modelli di veicolo (`params_vehicle.py`, `vehicle_14dof.py`), la trasmissione posteriore a doppio motore indipendente implementa 4 modalità matematiche selezionabili live:

1. **`DUAL_MOTOR_OPEN`**:
   Ripartizione simmetrica al 50/50 della coppia richiesta dal pilota:
   $$T_{RL} = T_{RR} = \frac{T_{\text{drive}}}{2}$$
   In presenza di slittamento asimmetrico rilevante ($\Delta\kappa > 0.12$), interviene la limitazione della coppia trasmessa basata sul grip della ruota con minore aderenza.

2. **`VIRTUAL_SPOOL`**:
   Differenziale bloccato al 100% (spool rigido). Reagisce alla differenza di velocità angolare delle ruote posteriori $\Delta\omega = \omega_{RL} - \omega_{RR}$ applicando una coppia di bloccaggio proporzionale:
   $$T_{\text{lock}} = K_{\text{spool}} \cdot (\omega_{RL} - \omega_{RR})$$
   $$T_{RL} = \frac{T_{\text{drive}}}{2} - T_{\text{lock}}, \qquad T_{RR} = \frac{T_{\text{drive}}}{2} + T_{\text{lock}}$$
   dove $K_{\text{spool}} = 80.0\text{ Nm/(rad/s)}$. Lo spool genera un forte momento stabilizzante (sottosterzante in ingresso, massima trazione in uscita).

3. **`VIRTUAL_SALISBURY_LSD`**:
   Differenziale a lamelle multi-disco con precarico statico $T_{\text{preload}} = 35\text{ Nm}$ e rampe asimmetriche in trazione (power: $\mu_{\text{power}} = 0.40$) e rilascio (coast: $\mu_{\text{coast}} = 0.20$):
   $$\mu_{\text{eff}} = \begin{cases} \mu_{\text{power}}, & T_{\text{drive}} \ge 10.0\text{ Nm} \\ \mu_{\text{coast}}, & T_{\text{drive}} < 10.0\text{ Nm} \end{cases}$$
   $$T_{\text{lock}} = (T_{\text{preload}} + \mu_{\text{eff}} |T_{\text{drive}}|) \cdot \tanh(3.0 \cdot (\omega_{RL} - \omega_{RR}))$$
   $$T_{RL} = \frac{T_{\text{drive}}}{2} - T_{\text{lock}}, \qquad T_{RR} = \frac{T_{\text{drive}}}{2} + T_{\text{lock}}$$
   La funzione liscia $\tanh$ garantisce regolarità differenziabile priva di discontinuità a $\Delta\omega \to 0$.

4. **`ACTIVE_TORQUE_VECTORING`**:
   Controllore d'imbardata in anello chiuso (Closed-Loop PI Yaw Rate Controller):
   - **Target Yaw Rate** da modello bicicletta lineare:
     $$r_{\text{lin}} = \frac{v_x}{L + K_{us} v_x^2} \cdot \delta_{\text{vis}}$$
     con gradiente di sottosterzo $K_{us} \approx 0.0018\text{ rad}/(\text{m/s}^2)$.
   - **Saturazione d'aderenza massima**:
     $$|r_{\text{target}}| \le \frac{\mu_{\text{tire}} \cdot g}{\max(v_x, 1.0)}$$
   - **Controllore PI con Anti-Windup**:
     $$e_r = r_{\text{target}} - r_{\text{curr}}$$
     $$\Delta M_z = \text{clip}\left(K_p e_r + K_i \int e_r dt, -\Delta M_{z,\max}, \Delta M_{z,\max}\right)$$
     con $K_p = 650.0\text{ Nm/(rad/s)}$, $K_i = 120.0\text{ Nm/rad}$ e $\Delta M_{z,\max} = 500\text{ Nm}$.
   - **Ripartizione di coppia differenziale**:
     $$\Delta T = \Delta M_z \cdot \frac{R_w}{t_r}$$
     $$T_{RL} = \text{clip}\left(\frac{T_{\text{drive}}}{2} - \Delta T, 0, T_{\max}\right), \qquad T_{RR} = \text{clip}\left(\frac{T_{\text{drive}}}{2} + \Delta T, 0, T_{\max}\right)$$
   - Sotto i $2\text{ m/s}$ ($7.2\text{ km/h}$), il sistema si disattiva dolcemente ripristinando la ripartizione simmetrica 50/50 a veicolo fermo o in manovra.

### 44.2 Interfaccia Ammortizzatori TTX25 & Registri 4-Way
Nel Tab *"Assetto & Aerodinamica"* del simulatore è integrato il controllo completo dell'idraulica:
- **Modello Idraulico**: Selettore tra `Lineare`, `Öhlins TTX25 Non-Lineare` e `Custom Dyno Curve`.
- **8 Registri Indipendenti**:
  - Asse Anteriore: LSC (0–25 click), HSC (0–4.0 giri), LSR (0–25 click), HSR (0–4.0 giri).
  - Asse Posteriore: LSC (0–25 click), HSC (0–4.0 giri), LSR (0–25 click), HSR (0–4.0 giri).
- **Tuning Thread-Safe e Ricompilazione Rapida**: La chiamata `update_live_tuning` aggiorna i parametri in modo atomico tramite `_tuning_lock` e ricompila la funzione CasADi fast ODE senza interruzioni nel thread di simulazione a 250–500 Hz.
- **Badge Smorzamento Modale ($\zeta$)**: Calcola istantaneamente i fattori di smorzamento adimensionale $\zeta_{\text{bump}}$, $\zeta_{\text{reb}}$ e $\zeta_{\text{avg}}$ a bassa velocità per asse anteriore e posteriore, con indicazione numerica dei coefficienti equivalenti $C_{\text{wheel}}$ in $\text{Ns/m}$.

### 44.3 3D Ghost Car & Live Delta-Lap Spaziale Continuo $\Delta t(s)$
- **Campionamento Traiettoria Spazio-Temporale**: Durante ogni giro cronometrato valido, il simulatore campiona ad alta risoluzione i nodi $(s, x, y, z, \psi, t_{\text{lap}})$ lungo la coordinata curvilinea $s$.
- **Archiviazione Best Lap**: Al passaggio sul gate del traguardo fotocellula (S/F line), se il giro è valido (nessun taglio a 4 ruote fuori pista per oltre $0.15\text{ s}$), il buffer viene archiviato in `best_lap_trajectory`.
- **Calcolo Delta Spaziale Continuo**: In ogni istante della simulazione lungo il tracciato alla posizione $s$:
  $$\Delta t(s) = t_{\text{current}} - t_{\text{best}}(s)$$
  Visualizzato in tempo reale nel card lap HUD in verde con segno negativo se la vettura è in vantaggio rispetto al tempo ideale, o in rosso con segno positivo se in ritardo.
- **Ghost Car 3D**: Interpolando temporalmente $x, y, z, \psi$ dal best lap all'istante $t_{\text{current}}$, il simulatore renderizza in OpenGL una ghost car tridimensionale satinata con canale alpha trasparente ($\alpha \approx 0.35$).

### 44.4 Ottimizzazione Display Lists OpenGL (`glGenLists`) & Pipelining Real-Time a 60–240 Hz
Per garantire fluidità costante a $60\text{ FPS}$ granitici (e fino a $120–240\text{ FPS}$ su display ad alto refresh rate):
- **Tracciato Stradale Precompilato**: L'intero nastro di asfalto, i cordoli dentati bicolore, la mezzeria tratteggiata e le linee di partenza/traguardo vengono compilati in una display list OpenGL hardware tramite `glGenLists(1)` e `glNewList(..., GL_COMPILE)`. Durante il render loop di `paintGL()`, una singola chiamata `glCallList` sostituisce decine di migliaia di vertici Python per-frame.
- **Scocca Veicolo 3D Precompilata**: In `vehicle_3d_models.py`, monoscocca, ali, prese d'aria, interni e roll hoop vengono compilati in una display list statica (`body_display_list`), ri-eseguita con una singola istruzione `glCallList`.
- **Gruppo Ruote e Pinze Freno Precompilate**: Il gruppo ruota rigido (battistrada Hoosier, spalle raccordate, labbro cerchio, canale ribassato, razze in lega, mozzo esagonale e disco freno ventilato) viene compilato separatamente per lato sinistro e destro in `wheel_display_list_left` e `wheel_display_list_right`. Le pinze freno fisse su upright vengono precompilate in `caliper_display_list_left` e `right`. Questa architettura abbatte oltre 11.500 chiamate C-types per frame a sole 8 veloci `glCallList`, riducendo il tempo di rendering ruote da $1.8\text{ ms}$ a $0.052\text{ ms}$ (35x più veloce).
- **Ghost Car 3D Istantanea**: La ghost car trasparente viene compilata in `ghost_display_list` ed eseguita con un'unica istruzione `glCallList`, riducendo il costo di disegno del best lap da $0.534\text{ ms}$ a $0.0229\text{ ms}$ (23x più veloce), eliminando ogni calo di FPS.
- **Eliminazione DWM VSync Stutter & SurfaceFormat `SwapInterval(0)`**: Configurando `QSurfaceFormat.setSwapInterval(0)` e invocando l'estensione WGL `wglSwapIntervalEXT(0)`, il driver grafico non blocca la GPU durante lo swap dei backbuffer. Questo previene il fenomeno per cui un frame leggermente oltre i $16.6\text{ ms}$ dimezzava improvvisamente la frequenza video a $28–30\text{ FPS}$, portando il throughput potenziale del render engine a oltre $400+\text{ FPS}$.
- **Timer Telematico ad Alta Risoluzione (`perf_counter`) & Target FPS**: La frequenza di emissione del worker fisico RK4 (250 Hz) verso la GUI adotta `time.perf_counter()` con intervallo programmabile `target_emit_dt` e selettore grafico (60 FPS, 120 FPS, 144 FPS, 240 FPS o Illimitato). Alla modalità nominale di $60\text{ FPS}$ (`target_emit_dt = 0.015 s`), ogni 4 step di fisica RK4 ($16.0\text{ ms}$) emette un frame esatto senza aliasing con i tick di sistema Windows ($15.6\text{ ms}$).

### 44.5 Restyling Modello 3D FSAE Low-Poly (3.000–8.000 Triangoli) & Supporto Wavefront `.obj`
Il generatore 3D della vettura Formula Student (`FSAECar3DModel`) in `vehicle_3d_models.py` include:
1. **Supporto File Mesh Esterno `.obj` con Fallback Trasparente**:
   - All'avvio, il modello ricerca la presenza di una mesh CAD esterna in `models/fsae_car.obj`, `assets/fsae_car.obj` o nella root.
   - Include un parser integrato `parse_obj_mesh` che legge vertici (`v`), normali (`vn`) e facce poligonali (`f`), triangola automaticamente gli n-goni mediante triangle-fan e calcola le normali di sfaccettatura ortogonali qualora assenti.
   - Se il file `.obj` non è presente, si attiva automaticamente il generatore procedurale avanzato.

2. **Geometria Procedurale Realistica ad Alto Dettaglio (Target Polycount Ottimizzato: ~2.590 Triangoli, -15.4% Riduzione Overhead)**:
   - **Monoscocca & Musetto**: Profilo affusolato a 7 sezioni aerodinamiche loftate, attenuatore d'impatto frontale (crashbox), fondo piatto integrale in fibra di carbonio e diffusore posteriore a espansione curva a 3 stadi con 4 paratie verticali (strakes).
   - **Cockpit Cavo & Interni**: Vasca interna con pianale ribassato, sedile a guscio in carbonio con seduta reclinata (~45°) e contenimenti laterali cosce/costole, cinture racing a 5 punti Sabelt con fibbia rotante metallica centrale e poggiatesta in schiuma ad alto assorbimento d'urto su paratia posteriore. Specchietti retrovisori aerodinamici in carbonio con vetro riflettente orientato verso il pilota.
   - **Roll-Hoop Tubolare 3D Volumetrico**: Arco principale realizzato con prisma regolare a 6 facce estruso lungo i punti di curvatura ($r = 13\text{ mm}$, `n_sides = 6`), puntoni diagonali posteriori di rinforzo verso il telaio motore e roll-hoop anteriore di sicurezza sotto il cruscotto.
   - **Sidepod Undercut Coke-Bottle**: Fiancate aerodinamiche con imbocco frontale radiatori, pacco radiante metallico arretrato, svasatura inferiore (undercut) per alimentare il diffusore e branchie louvered superiori di evacuazione termica.
   - **Aerodinamica Multi-Elemento**: Ala anteriore a 8 sezioni di apertura alare con profilo principale cambered, flap superiore con Gurney flap, paratie verticali sagomate con footplate sigillante ed estensione laterale; ala posteriore biplana ad alto carico con paratie forate da louvers di scarico pressione, piloni curvi swan-neck a 6 facce a presa superiore e beam wing inferiore.
   - **Ruote Solide Hoosier 16x7.5-10 Ottimizzate**: 22 settori radiali (`n_seg = 22`), battistrada cilindrico in gomma opaca, spalle raccordate interne ed esterne, fianco rastremato verso il tallone, labbro cerchio tornito lucido, canale interno ribassato (drop center), 5 razze solide in lega leggera, dado centrale esagonale, disco freno in acciaio con fascia ventilata e pinza freno anodizzata.
   - **Sospensioni Tubolari 3D Volumetriche**: Triangoli A-arm superiori e inferiori realizzati con tubi esagonali 3D (`_draw_tube_strut`, `n_sides = 6`), pushrod diagonale in titanio satinato, tirante di sterzo e montante ruota in lega ricavato dal pieno.
   - **Volante a Farfalla FSAE & Cruscotto Digitale LCD**: Impugnature ergonomiche in gomma/Alcantara, piastra centrale in carbonio con incavi per le dita, display LCD da corsa (indicatore marcia "1", contagiri grafico a barre RPM a 3 zone verde/giallo/rosso, telemetria), strip superiore di 8 LED shift lights multicolore e pulsanti di controllo (DRS, limitatore pit, radio). Ruota dinamicamente attorno all'asse del piantone inclinato di 18° in base all'angolo di sterzata `steer_deg`.

3. **Calibrazione Camera Cockpit POV Pilota (`interactive_driver_14dof.py`)**:
   - Coordinate punto di vista pilota in asse vettura: `loc_x = 0.22`, `loc_y = 0.0`, `loc_z = 0.46`.
   - Piano di clipping vicino (`near_clip`): impostato a $0.02\text{ m}$ ($2\text{ cm}$) in `gluPerspective(72.0, aspect, 0.02, 1500.0)`.
   - Visuale in perfetta continuità: inquadra in primo piano la corona superiore del volante a farfalla con i LED di cambiata, lo sguardo oltre il volante mostra la sommità del musetto rastremato, i puntoni delle sospensioni e le ruote anteriori sterzanti, senza compenetrazioni o tagli visivi della geometria.

### 44.6 Monitor Frame Rate Video (FPS Counter) & Performance Profiling
Nel simulatore real-time `interactive_driver_14dof.py` è integrato un sistema di monitoraggio del frame rate del rendering OpenGL:
- **Misura & Filtraggio Esponenziale**: In `paintGL()`, il delta temporale inter-frame $\Delta t_{\text{frame}}$ viene misurato ad alta precisione con `time.perf_counter()` e filtrato tramite media mobile esponenziale ($\text{FPS}_{\text{smooth}} = 0.90 \cdot \text{FPS}_{\text{smooth}} + 0.10 \cdot \text{FPS}_{\text{inst}}$) per garantire una lettura stabile e priva di sfarfallio.
- **Badge Dedicato Top-Right**: Un badge ad alto contrasto posizionato accanto al pulsante fullscreen visualizza istantaneamente i frame al secondo (`XX FPS`) con codifica cromatica (verde per $\ge 55\text{ FPS}$, giallo per $30–54\text{ FPS}$, rosso per $< 30\text{ FPS}$).
- **Integrazione Card Prestazioni & Sidebar**: Il valore di rendering FPS è affiancato alla frequenza di integrazione fisica RK4 (`⚡ RK4: 250 Hz | XX FPS`) nella card overlay HUD e nel pannello laterale del monitor real-time.

---

## 45. Calibrazioni Fisiche ET-18, ARB Diamond & Risoluzione Dinamica Stabilità

### 45.1 Powertrain Dual Motor RWD & Modello Accelerazione Quasi-Statica (75m Sprint)
Nei prototipi Formula Student moderni ad alte prestazioni (come la configurazione ET-18), la trazione posteriore adotta due motori elettrici indipendenti (es. 2x EMRAX / AMK) montati in asse o rinvio con riduzione fissa $u = 4.0:1$:
- **Momento Torcente alla Ruota**:
  Ciascun motore sviluppa una coppia nominale massima $T_{\text{motor}} = 100\text{ Nm}$. Con il rapporto di trasmissione $u = 4.0$, la coppia massima erogabile alla singola ruota motrice è:
  $$T_{\text{wheel,single}} = T_{\text{motor}} \cdot u \cdot \eta = 100 \cdot 4.0 \cdot 0.95 = 380\text{ Nm}$$
  (con picco teorico a $400\text{ Nm}$). L'intero asse posteriore motrice riceve fino a:
  $$T_{\text{axle,tot}} = 2 \cdot T_{\text{motor}} \cdot u = 800\text{ Nm}$$
  generando una forza di spinta longitudinale alla patch di contatto di circa:
  $$F_{x,\text{motor}} = \frac{T_{\text{axle,tot}}}{R_w} = \frac{800}{0.228} \approx 3508\text{ N}$$
- **Limitazione di Potenza Regolamentare (80 kW)**:
  Alle velocità elevate ($v > 22.8\text{ m/s} \approx 82\text{ km/h}$), la coppia motore viene modulata per non superare il tetto massimo del regolamento FSAE $P_{\max} = 80\text{ kW}$:
  $$F_{x,\text{power}} = \frac{P_{\max} \cdot \eta}{v_x}$$
- **Aderenza Longitudinale Reale**:
  Sul treno di pneumatici Hoosier 16x7.5-10 R20, il coefficiente di attrito longitudinale $\mu_x$ raggiunge valori di picco tra $1.55$ e $1.65$ g. Con il corretto bilancio di coppia dual motor, l'accelerazione massima iniziale $a_{x,\max}$ sale a $1.16–1.25\text{ g}$, e il tempo sui $75\text{ m}$ scende dai valori fittizi di $4.75\text{ s}$ al tempo reale da competizione:
  $$t_{75m} \approx 3.92\text{ s}, \quad v_{\text{final}} \approx 101.4\text{ km/h}$$

### 45.2 Vertici Rombo ARB da Curve Digitalizzate (05 Ideale & 06 Reale)
La mappa "Rombo ARB" (Diamond Balance) correla la rigidezza di rollio totale $K_{\phi,\text{tot}} = K_{\phi,f} + K_{\phi,r}$ e la ripartizione anteriore $R_f = K_{\phi,f} / K_{\phi,\text{tot}}$. Nel modulo `mdo_arb_diamond_balance.py`:
- **Curva 05: Ideale (Telaio Infinitamente Rigido)**: Vertici calcolati presupponendo rigidezza torsionale del telaio infinita ($K_{\text{chassis}} = \infty$), derivati dai punti estremi delle posizioni lama ARB digitalizzate.
- **Curva 06: Reale (Cedevolezza Torsionale Inclusa)**: Vertici calcolati inserendo la rigidezza torsionale finita del telaio monoscocca in serie:
  $$\frac{1}{K_{\phi,\text{eff}}} = \frac{1}{K_{\phi,\text{arb+spring}}} + \frac{1}{K_{\text{torsion}}}$$
- I preset sono memorizzati in `setups/arb_diamond/` e possono essere richiamati con un click nella GUI (`main_gui_classic.py`).

### 45.3 Analisi di Stabilità Dinamica: Trattamento Modi Aperiodici Instabili ($\det(A) \le 0$)
Dalla matrice di stato linearizzata nel piano $(\beta, r)$:
$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}, \quad \det(A) = \frac{L^2 C_{\alpha f} C_{\alpha r}}{m I_z v_x^2} \left(1 + \frac{K_{us} v_x^2}{L}\right)$$
- **Regime Oscillatorio Stabile**: Se $\det(A) > 0$ e $\text{tr}(A) < 0$, gli autovalori sono complessi coniugati con parte reale negativa. La frequenza naturale e lo smorzamento sono definiti da:
  $$\omega_n = \sqrt{\det(A)}, \quad f_n = \frac{\omega_n}{2\pi}, \quad \zeta = -\frac{\text{tr}(A)}{2\omega_n}$$
- **Regime Aperiodico Instabile ($v_x > v_{\text{crit}}$)**: Per un veicolo sovrasterzante ($K_{us} < 0$), sopra la velocità critica $v_{\text{crit}} = \sqrt{L / (-K_{us})}$, si ha $\det(A) \le 0$. Gli autovalori diventano reali con segni opposti ($\lambda_1 > 0$, $\lambda_2 < 0$), indicando una divergenza pura di imbardata (spin-out).  
  In questa condizione non esiste un moto armonico oscillatorio: il software restituisce formalmente `f_n_hz = None`, `omega_n = None`, `zeta = None` e descrive il modo come `"Aperiodico (Instabile)"`, eliminando la fuorviante visualizzazione di $0.00\text{ Hz}$.

### 45.4 Geometria Monoscocca & Penalità Quadratica Normalizzata
Per evitare che l'ottimizzatore 14-DOF espanda artificialmente passo e carreggiate fino ai limiti massimi consentiti dal regolamento ($L \to 1.80\text{ m}$, $t \to 1.50\text{ m}$), le dimensioni geometriche del monoscocca sono mantenute fisse per impostazione predefinita nel catalogo ET-18 ($L = 1.529\text{ m}$, $t_f = 1.252\text{ m}$, $t_r = 1.136\text{ m}$).  
Qualora l'utente decida di abilitarne l'ottimizzazione, la funzione obiettivo applica una penalità quadratica normalizzata rispetto al telaio nominale:
$$\text{geom\_penalty} = \text{penalty\_factor} \cdot 5.0 \cdot \left[ \left(\frac{L - L_0}{L_0}\right)^2 + \left(\frac{t_f - t_{f0}}{t_{f0}}\right)^2 + \left(\frac{t_r - t_{r0}}{t_{r0}}\right)^2 \right]$$

### 45.5 Architettura Console Solutore: QSplitter & Popout Sincronizzato
Nella nuova GUI (`main_gui_pro.py`), il pannello ispettore integra:
- **`QSplitter` Verticale**: Consente il ridimensionamento fluido trascinando la barra di divisione verso l'alto o verso il basso, eliminando il limite rigido a 110 pixel.
- **Finestra Popout Dedicata (`ConsolePopoutDialog`)**: Condivide il medesimo `QTextDocument` in memoria, garantendo sincronizzazione bidirezionale a zero latenza con conteggio righe in tempo reale, ricerca testuale e svuotamento rapido.

---

## 46. Simmetria Dinamica di Rettilineo, Specchiatura Slittamento Combinato & Cache Binaria C

### 46.1 Invarianza di Rettilineo & Diagnosi Forze Laterali Asimmetriche
In condizioni di marcia rettilinea su fondo piano uniforme ($\delta = 0$, $v_y = 0$, $r = 0$, profilo stradale $z_{road} = 0$), le leggi di conservazione della quantità di moto e del momento angolare impongono:
$$\Sigma F_y = F_{y,fl} + F_{y,fr} + F_{y,rl} + F_{y,rr} \equiv 0.00\text{ N}$$
$$\Sigma M_z = a (F_{y,fl} + F_{y,fr}) - b (F_{y,rl} + F_{y,rr}) + \frac{t_f}{2}(F_{x,fl} - F_{x,fr}) + \frac{t_r}{2}(F_{x,rl} - F_{x,rr}) \equiv 0.00\text{ Nm}$$
Durante la pura accelerazione a farfalla aperta, qualsiasi sbilanciamento di segno o modulo in $F_y$ tra ruota sinistra e ruota destra genera una deviazione parassita di traiettoria ($\dot{v}_y \ne 0$, $\dot{r} \ne 0$) e induce forze apparenti errate nel feedback al pilota.

La risoluzione del problema ha richiesto due interventi strutturali congiunti:
1. **Specchiatura dei coefficienti di slittamento combinato Pacejka**: Nel modello MF6.1 i termini $RVY_1$, $RVY_2$, $RVY_3$ e $RHX_1$ modellano effetti di ply-steer e accoppiamento longitudinale/laterale. Senza l'applicazione dei fattori $\lambda_{Vy}$ e $\lambda_{Hy}$ a seconda del lato della ruota (determinato dalla chiave `TYRESIDE` del file `.tir`), l'accelerazione a slip elevato ($\kappa > 0$) causava uno shift della forza laterale $S_{Vy\kappa}$ concorde su entrambe le ruote posteriori. Con i moltiplicatori $\lambda_{Vy}, \lambda_{Hy}$, le forze sono esattamente uguali in modulo e opposte in verso ($F_{y,rl} = -F_{y,rr}$).
2. **Allineamento della convenzione di convergenza (Toe)**: Nel file parametri (`params_vehicle.py`), `toe_r_static` è definito come convergenza positiva (**toe-in**). L'inversione di segno nei modelli 14-DOF, 7-DOF e 3-DOF trasformava il setup in divergenza (**toe-out**), generando una forza d'angolo contrapposta alla spinta di camber che spingeva la ruota sinistra verso l'esterno. Con il segno corretto ($\alpha_{rl} = \dots - \delta_{toe,r}$, $\alpha_{rr} = \dots + \delta_{toe,r}$), camber thrust e toe thrust collaborano per spingere entrambe le ruote verso la mezzeria con simmetria antisimmetrica perfetta.

### 46.2 Invalidazione Dinamica della Cache Binaria C (`get_model_signature`)
Il simulatore 14-DOF e i solutori ad alta velocità compilano il grafo ODE CasADi in C/GCC (`ode_14dof_<hash>.dll`) per garantire frequenze di calcolo $\ge 500\text{ Hz}$ a bassa latenza:
- **Hashing SHA-256**: La firma del modello (`Vehicle14DOF.get_model_signature()`) concatena i parametri di massa, inerzia, geometria delle sospensioni, rigidezze, aerodinamica, i parametri di camber/toe statico (`camber_f_static`, `camber_r_static`, `toe_f_static`, `toe_r_static`), la chiave `TYRESIDE` e un salt di versione strutturale (`'v3_combined_mirror_toe'`).
- **Prevenzione Binari Obsoleti**: Se il codice simbolico o i parametri di assetto cambiano, la firma genera un nuovo hash univoco a 16 caratteri esadecimali. Il sistema rileva automaticamente l'assenza della nuova DLL compilata, genera il file sorgente `.c` tramite CasADi Code Generator, richiama `MinGW GCC` con ottimizzazione `-O3` e carica la libreria aggiornata a caldo, eliminando il rischio di eseguire cache binarie non allineate.

### 46.3 Suite di Test di Simmetria Rettilinea (`tests/test_straight_line_symmetry.py`)
È stata integrata nella suite di continuous integration una serie di 7 test di verifica automatica:
1. `test_tyreside_detection`: Verifica la lettura corretta di `TYRESIDE` da `.tir` (`RIGHT` per Hoosier 43075, `LEFT` per SC24).
2. `test_14dof_straight_line_symmetry`: Verifica che a veicolo fermo o a velocità costante con $\delta = 0$, le forze laterali soddisfino $|F_{y,fl} + F_{y,fr}| < 10^{-3}\text{ N}$, $|F_{y,rl} + F_{y,rr}| < 10^{-3}\text{ N}$ e $\dot{r} < 10^{-5}\text{ rad/s}^2$.
3. `test_14dof_camber_thrust_signs`: Verifica che con camber negativo statico, la ruota sinistra spinga a destra ($F_{y,left} < 0$) e la destra a sinistra ($F_{y,right} > 0$).
4. `test_14dof_acceleration_combined_slip_symmetry`: Verifica la simmetria a piena accelerazione con $\kappa = 0.05$.
5. `test_interactive_driver_telemetry_symmetry`: Verifica che le forze lette dalla telemetria del simulatore real-time rispecchino fedelmente la simmetria.
6. `test_7dof_straight_line_symmetry`: Verifica l'invarianza del modello 7-DOF.
7. `test_3dof_straight_line_symmetry`: Verifica l'invarianza del modello 3-DOF.

---

## 47. Modello Propulsivo Elettrico ad Alta Fedeltà: Motore EMRAX 188HV LC, Rapporto 4:1, Mappa Efficienza 2D & Limite Accumulatore 80 kW (FSAE EV)

La suite implementa un modello fisico-numerico completo del gruppo motopropulsore elettrico in [`src/models/powertrain_electric.py`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/src/models/powertrain_electric.py), calibrato rigorosamente sulle curve del costruttore (**EMRAX 188HV Liquid-Cooled**) contenute nella cartella `data/motor_curve/` e vincolato dalle norme ufficiali del regolamento **Formula SAE / Formula Student EV**.

### 47.1 Specifiche Motore EMRAX 188HV LC & Curve di Picco (`01_power_torque.png`)
Il motore è configurato per funzionare sulle curve di picco transitorie (**Peak Power** e **Peak Torque**):
- **Coppia di Picco ($T_{\text{max,peak}} = 100.0\text{ Nm}$)**:
  - Rampa iniziale $0-500\text{ RPM}$ da $96\text{ Nm}$ a $100\text{ Nm}$.
  - Ampio plateau a coppia costante $T = 100.0\text{ Nm}$ esteso da $500$ a $5500\text{ RPM}$.
  - Decadimento in regime di deflussaggio (constant power envelope) verso $70.0\text{ Nm}$ a $8000\text{ RPM}$.
  - Cutoff netto a zero al superamento del regime massimo consentito ($n_{\text{max}} = 8000\text{ RPM}$).
- **Potenza Meccanica di Picco ($P_{\text{mech,peak}} \approx 60.0\text{ kW}$)**:
  - Il picco di potenza del singolo motore si raggiunge nell'intervallo $6200-6500\text{ RPM}$ ($\approx 60.0\text{ kW}$ meccanici a $6500\text{ RPM}$ con $T \approx 88.2\text{ Nm}$).
- **Approssimazione Simbolica CasADi Liscia (`casadi_peak_torque`)**:
  Per l'integrazione nei solutori NLP basati su derivate (IPOPT / MUMPS) e nella Direct Collocation, la curva di coppia di picco è formulata in forma differenziabile continua senza discontinuità $C^0$:
  $$T_{\text{smooth}}(n) = T_{\text{plateau}} - \frac{1}{k}\ln\left(1 + \exp\left[k(T_{\text{plateau}} - T_{\text{power\_lim}}(n))\right]\right)$$
  con $T_{\text{power\_lim}}(n) = \frac{60000}{\omega_m(n)}$, parametro di raccordo $k = 0.08$ e cutoff sigmoideo morbido centrato a $8200\text{ RPM}$:
  $$\text{cutoff}(n) = \frac{1}{2}\left[1 + \tanh\left(\frac{8200 - n}{150}\right)\right]$$

### 47.2 Cinematica di Trasmissione & Riduzione 4:1 Dual-Motor
La catena cinematica posteriore adotta due motori indipendenti montati sul telaio (uno per la ruota posteriore sinistra RL, uno per la ruota posteriore destra RR):
- **Rapporto di Trasmissione Fisso**: $i = 4.0$ (la ruota gira $4$ volte più lentamente dell'albero motore):
  $$\omega_{\text{motor},i} = 4.0 \cdot \omega_{\text{wheel},i}, \quad n_{\text{RPM},i} = \omega_{\text{motor},i} \cdot \frac{60}{2\pi}$$
- **Moltiplicazione di Coppia alla Ruota**:
  $$T_{\text{wheel},i} = T_{\text{motor},i} \cdot i \cdot \eta_{\text{drivetrain}}$$
  con rendimento meccanico della trasmissione a cascata/catena $\eta_{\text{drivetrain}} = 0.95$. A $100\text{ Nm}$ di picco motore corrispondono $380\text{ Nm}$ alla ruota ($760\text{ Nm}$ complessivi sull'asse posteriore).

### 47.3 Mappa di Efficienza Motore 2D $\eta(\text{RPM}, T)$ (`02_efficiency_speed.png`)
Il rendimento del motore varia nello spazio operativo bidimensionale $(n, T)$ ed è gestito da un interpolatore bilineare a griglia regolare (`RegularGridInterpolator`, $17 \times 11$ nodi):
- **Sweet Spot Island ($\eta \ge 96\%$)**:
  Isola ellittica centrata a $3800\text{ RPM}$ e $44\text{ Nm}$ con rendimento di picco $\eta_{\text{max}} = 96.6\%$, estesa nell'intervallo $2300-5700\text{ RPM}$ e $30-57\text{ Nm}$.
- **Contorno al 95%**: $1900-6700\text{ RPM}$ e $25-69\text{ Nm}$.
- **Contorno al 94%**: $1600-8000\text{ RPM}$ e $22-80\text{ Nm}$.
- **Contorni al 90-94% e 86-90%**: coprono la maggior parte dell'inviluppo dinamico fino a $8000\text{ RPM}$. A bassissima coppia o velocità il rendimento degrada in modo continuo verso il valore limite inferiore calibrato a $0.50$.

### 47.4 Regolamento FSAE EV: Tetto di Potenza Elettrica Accumulatore a 80 kW
Conformemente al regolamento ufficiale **Formula SAE EV Rules (Energy Meter Limitation)**, il vincolo di potenza massima si applica rigorosamente alla **potenza elettrica totale prelevata dall'accumulatore**:
$$P_{\text{elec,total}} = P_{\text{elec,RL}} + P_{\text{elec,RR}} \le 80.0\text{ kW}$$
Poiché $P_{\text{mech},i} = T_{\text{motor},i} \cdot \omega_{\text{motor},i}$ e $P_{\text{elec},i} = \frac{P_{\text{mech},i}}{\eta_i(n_i, T_{\text{motor},i})}$, la massima potenza meccanica erogabile alle ruote dipende in ogni istante dall'efficienza puntuale dei motori.

#### Gestione Asimmetrica del Torque Vectoring con Potenza Satùra:
Quando la richiesta di trazione eccede la capacità del pacco batterie ($P_{\text{elec,RL}} + P_{\text{elec,RR}} > 80.0\text{ kW}$), il modulo [`allocate_dual_motor_torques`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/src/models/powertrain_electric.py) ripartisce la potenza elettrica disponibile preservando fedelmente il differenziale di momento di imbardata richiesto dal controllore ($\Delta P = P_{\text{elec,RR}} - P_{\text{elec,RL}}$):
$$P_{\text{elec,RR}}^{\text{tgt}} = \frac{80.0\text{ kW} + \Delta P}{2}, \quad P_{\text{elec,RL}}^{\text{tgt}} = \frac{80.0\text{ kW} - \Delta P}{2}$$
Le potenze dei singoli motori vengono quindi saturate alle massime capacità fisiche individuali $[0, P_{\text{elec,max},i}(n_i)]$. Questo permette, a titolo di esempio, un'erogazione asimmetrica di **$50\text{ kW}$ sulla ruota esterna e $30\text{ kW}$ su quella interna**, fornendo pieno supporto al sovrasterzo/sottosterzo controllato in curva pur mantenendo il consumo istantaneo sotto la soglia di squalifica dell'Energy Meter.

### 47.5 Integrazione Nei Modelli Fisici & GUI
- **Modello 3-DOF Steady-State (`src/models/vehicle_3dof.py`)**:
  Metodi `get_tractive_envelope(v_x)` e `evaluate_operating_point(v_x, Fx_req)` per ricavare la forza di trazione massima ammissibile tenendo conto dell'inviluppo EMRAX 188, della mappa di efficienza e del tetto a 80 kW.
- **Modelli 7-DOF e 14-DOF (`src/models/vehicle_7dof.py`, `src/models/vehicle_14dof.py`)**:
  Metodo `apply_powertrain_limits(w_rl, w_rr, T_req_rl, T_req_rr)` applicato direttamente nel calcolo delle derivate di spin delle ruote $\dot{\omega}_i$.
- **Simulatore Real-Time (`interactive_driver_14dof.py`)**:
  Il ciclo fisico a $500\text{ Hz}$ esegue la saturazione e l'allocazione real-time. L'HUD su schermo integra un pannello dedicato `pt_group` con monitor della potenza elettrica totale (barra colorata $0-80\text{ kW}$ con allerta saturazione), potenza meccanica, giri motore RL/RR e rendimento istantaneo.
- **Simulatore di Accelerazione Longitudinale (`mdo_acceleration.py`)**:
  Integrazione delle curve di coppia e dell'Energy Meter sul test dei 75 metri FSAE, con calcolo automatico dell'energia consumata in Wh (meccanica ed elettrica) e dell'efficienza energetica media del run.
- **Eventi Dinamici (`mdo_dynamic_events.py`)**:
  Le manovre Step Steer, Slalom, DLC ed Endurance integrano l'allocazione di coppia realistica con rapporto 4:1 e limite ad 80 kW.

---

## 15. Standardizzazioni Fisiche & Architettura V4.7.1

### 15.1 Single Source of Truth (SSOT) Massa Veicolo (292.0 kg)
Tutti i modelli fisici (3-DOF, 7-DOF, 14-DOF, MDO, Simulatore Real-Time) adottano la massa nominale di $292.0\text{ kg}$ come configurazione pronta corsa con pilota a bordo inclusa in `VehicleParams.m`. È stata rimossa ogni addizione duplicata di massa pilota (`m_driver`) nei calcoli dinamici e nei badge telemetrici.

### 15.2 Accoppiamento Rigidezze di Rollio & LLTD negli Sweep 1D
Nelle analisi di sensibilità parametrica (`mdo_sensitivity.py` e `transient_sensitivity.py`), la variazione di $K_{\phi,f}$ o $K_{\phi,r}$ aggiorna istantaneamente la ripartizione del carico laterale elastico:
$$LLTD_{\text{elastic}} = \frac{K_{\phi,f}}{K_{\phi,f} + K_{\phi,r}}$$
propagando il valore su `vp.lltd` per garantire coerenza con tutti i solutori stazionari e di stabilità. Inoltre, sono supportati sweep parametrici indipendenti su `lltd` (a rigidezza complessiva costante) e su `roll_stiffness_tot` (a ripartizione costante).

### 15.3 Sensibilità Transitoria & Centri di Rollio nell'Interfaccia PRO
Il modulo di sensibilità transitoria e analisi centri di rollio è stato integrato stabilmente nel **Workspace 3: Ottimizzazione MDO & Sensibilità** come `Sensibilità Transitoria & RC`, con mapping reattivo `_navigate_to_widget` e rimozione dei riferimenti legacy `(Tip #28, #30)`.

### 15.4 Sblocco Coppia Dual Motor nel Simulatore 14-DOF
È stato eliminato il filtro TCS statico su $\kappa > 0.10$ che limitava artificialmente la coppia a 50 Nm e l'accelerazione a 0.46 g. Con l'allocazione fisica EMRAX 188HV / 80 kW, ciascun motore posteriore eroga fino a $100\text{ Nm}$ ($400\text{ Nm}$ per ruota, $800\text{ Nm}$ all'asse), consentendo accelerazioni di picco realistiche $a_x > 1.0-1.2\text{ g}$.

### 15.5 Ottimizzatore Direct Collocation & Racing Line De-conservativo (NLP CasADi + IPOPT)
- **Visualizzazione ad Alto Contrasto & Tema Scuro**: Integrazione completa con `ui_theme_engine.py` nel dialogo `run_direct_collocation_dialog`: testi, etichette e tabelle con palette ad alta visibilità (`#f0f6fc` / `#E0E0E0`), bordi (`#30363d`), centerline tratteggiata in grigio chiaro (`#888888`) al posto del nero invisibile (`'k:'`), e richiamo automatico di `apply_plot_theme(fig, axes)`.
- **De-conservativismo Formulazione OCP**:
  - Allargamento dell'inviluppo di aderenza utilizzabile (fino al 98%–99% del grip pneumatico).
  - Estensione del limite superiore di velocità a $45.0\text{ m/s}$ ($162\text{ km/h}$).
  - Allentamento box bounds su deviazione angolare ($\xi \in [-42^\circ, +42^\circ]$) e angolo di sterzo ($\delta \in [-28^\circ, +28^\circ]$).
  - Regolarizzazione sui tassi di controllo (jerk minimization): sostituzione della penalizzazione quadratica sull'ampiezza di controllo con penalità sui differenziali finiti $\sum (\Delta a_{x,\text{bar}})^2$ e $\sum (\Delta \delta_{\text{bar}})^2$ pesati a $10^{-6}$ (e termine convessificante minimo $10^{-7} \sum a_{x,\text{bar}}^2$), consentendo l'uso pieno della trazione e della frenata.
  - Doppio vincolo di trazione: limite combinato di coppia motrice ai bassi regimi ($F_{x,\text{trac}} \le \frac{n_{\text{motors}} T_{\max} \tau_{\text{gear}} \eta}{R_w}$) e di potenza elettrica di picco ($F_{x,\text{trac}} \cdot v \le P_{\max}$).

### 15.6 Legge di Contatto al Suolo Unilaterale & Stabilità Multi-Rate 500-1500 Hz (14-DOF)
- **Modello di Contatto Unilaterale**: Sostituzione della formula tensiva con legge unilaterale $F_{z,i} = \min\left( \text{gate}_i \cdot \max(0, K_{tz}\delta_i + C_{tz} v_{z,i}), F_{z,\max} \right)$ con $\text{gate}_i = 0.5(1 + \tanh(500 \delta_i))$ e $\delta_i = \delta_{0,i} + z_i$, eliminando qualsiasi suzione/trazione verso il suolo e preservando la convergenza RK4 del 4° ordine (nessun clamp non-differenziabile su $\dot{v}_z$).
- **Velocità di Trasporto Patch di Contatto**: Incorporamento della velocità periferica di spin $|\omega_i| R_w$ in $V_{\text{transport},i} = \max(\sqrt{V_{xw}^2 + V_{yw}^2}, |\omega_i| R_w)$ con soglia minima $\epsilon_v = 0.8\text{ m/s}$, impedendo l'allungamento sproporzionato del tempo di rilassamento in partenze da fermo a pieno gas e azzerando il ciclo limite instabile a 22 Hz.
- **Accoppiamento Beccheggio & Anti-Squat**: Bounding dei polinomi cinematici di Adams Car e clamping del fattore di beccheggio destabilizzante `anti_factor` nell'intervallo $[0.20, 1.40]$.
- **Selettore Multi-Rate 500 Hz / 1000 Hz / 1500 Hz**: Supporto a frequenze di integrazione fino a 1500 Hz nel simulatore interattivo real-time, con signature C DLL continua e array contigui `float64` per ctypes.

### 15.7 Mesh di Collisione 3D Allargata a 5 Strisce & Motore di Rugosità Stradale ISO 8608
- **Mesh di Collisione 3D a 5 Strisce (Verde, Cordolo, Asfalto, Cordolo, Verde)**:
  - Generazione di 5 strisce di quadrilateri contigue e prive di discontinuità geometrica lungo tutto il circuito 3D:
    - Striscia 1: Verde Sinistro ($w_{\text{verge}} = 15.0\text{ m}$, `surface_type = 2`), segue l'inclinazione di banking $b(s)$ con pendenza di drenaggio graduale ($h_{\text{drop}} = 0.15\text{ m}$).
    - Striscia 2: Cordolo Sinistro ($w_{\text{kerb}} = 0.80\text{ m}$, `surface_type = 1`), segue l'inclinazione di banking $b(s)$ con smussatura/rialzo di $+15\text{ mm}$ sul piano stradale.
    - Striscia 3: Asfalto Pista (`surface_type = 0`, tra bordo sinistro e destro).
    - Striscia 4: Cordolo Destro ($w_{\text{kerb}} = 0.80\text{ m}$, `surface_type = 1`, $+15\text{ mm}$).
    - Striscia 5: Verde Destro ($w_{\text{verge}} = 15.0\text{ m}$, `surface_type = 2`, $-0.15\text{ m}$).
  - Ampiezza totale del corridoio di collisione estesa da $3.8\text{ m}$ a $\approx 35.4\text{ m}$.
  - Chiusura toroidale stagna per piste chiuse ($N-1 \to 0$), eliminando qualsiasi gap geometrico in corrispondenza del traguardo (Start/Finish).
  - Algoritmo di Fallback Analitico Continuo: qualora una traiettoria anomala porti il veicolo oltre i $15\text{ m}$ di verde, la quota e la normale al suolo vengono proiettate analiticamente sul piano inclinato della stazione curvilinea più vicina con normale inclinata $\mathbf{n} = (-\sin(b)\mathbf{n}_{\text{left}} + \cos(b)\mathbf{k})$ e `surface_type = 2`, eliminando il crollo della quota al baricentro vettura e il collasso della normale a $[0, 0, 1]$.
- **Motore di Rugosità Stradale ISO 8608**:
  - Classi ISO 8608 implementate:
    - Classe A: $G_d(n_0) = 4.0 \times 10^{-6}\text{ m}^3, \sigma_z = 1.41\text{ mm}$ (Autodromo internazionale)
    - Classe B: $G_d(n_0) = 16.0 \times 10^{-6}\text{ m}^3, \sigma_z = 2.83\text{ mm}$ (Buono / Stradale)
    - Classe C: $G_d(n_0) = 64.0 \times 10^{-6}\text{ m}^3, \sigma_z = 5.66\text{ mm}$ (Medio / Usurato)
    - Classe D: $G_d(n_0) = 256.0 \times 10^{-6}\text{ m}^3, \sigma_z = 11.31\text{ mm}$ (Sconnesso / Buche)
    - Disattivato (Piano): $G_d(n_0) = 0.0, \sigma_z = 0.0\text{ mm}$ (Liscio)
  - Architettura di contatto a due livelli: la mesh collisione macroscopicamente gestisce banking e rilievi 3D a basso costo computazionale; il modulo `RoadSurface2D` esegue l'interpolazione bilineare sub-microsecondo della rugosità ISO e calcola la derivata temporale $\dot{z}_{\text{road}} = \frac{\partial Z}{\partial s} \cdot v_x$, applicata agli smorzatori TTX25 per un'eccitazione idraulica e haptic realistica senza aliasing da triangolazione.
- **Badge HUD Non-Interferente & Monitor 4 Impronte**:
  - Badge sintetico posizionato nella barra superiore (`x = 232, y = 12, w = 245, h = 54`), adiacente al Lap Timer e totalmente disaccoppiato dalle forze ruota.
  - Monitor dedicato `MICRO-RUGOSITÀ FONDO (4 IMPRONTE)` posizionato direttamente sopra la matrice `ASSETTO & GOMME` (`x = 14, y = mat_y - rough_h - 8, w = 295, h = 58`), eliminando qualsiasi sovrapposizione grafica e disambiguando con precisione i valori di deflessione microscopica sotto ciascuna ruota (ANT SX, ANT DX, POST SX, POST DX) unitamente alla segnalazione del filtro d'impronta 2D ($140 \times 160\text{ mm}$).
- **Proiezione Curvilinea Continua Frenet-Serret & Smoothing Telecamera**:
  - Proiezione analitica continua del veicolo sul segmento di mezzeria tracciato: $s_{\text{curr}} = s_{\text{tr}}[i] + \text{proj} \cdot \Delta s_{\text{seg}}$ e $d_{\text{car}} = -\Delta x \sin\theta + \Delta y \cos\theta$. Eliminata la discretizzazione a scatti al waypoint più vicino che generava onde quadre a 20 Hz su quota $z_{\text{road}}$, pendenza e banking.
  - Filtro esponenziale (EMA) armonizzato tra coordinate orizzontali e verticali: `smooth_z`, `smooth_grade`, `smooth_bank` aggiornati con il medesimo ritardo di fase di `smooth_x` e `smooth_y`, garantendo zero sfasamento tra telaio e asfalto.
  - Smooth Bank Follow per il vettore `up` della camera chase (40% lean verso la normale inclinata della curva sopraelevata), eliminando lo shear torsionale e il roll jitter in gluLookAt.
- **Filtro Enveloping Impronta 2D & Smorzamento Carcassa Pneumatico**:
  - Filtro bidimensionale separabile di Hanning applicato sia lungo la coordinata longitudinale $s$ ($L_{\text{patch}} = 140\text{ mm}$) che lungo la coordinata laterale $d$ ($W_{\text{patch}} = 160\text{ mm}$), simulando il bridging geometrico del battistrada reale sopra gli aggregati di asfalto e abbattendo i gradienti spaziali $\partial Z / \partial s$ non fisici.
  - Rilassamento dinamico della carcassa del pneumatico: $\tau_{\text{patch}} = \frac{L_{\text{patch}}}{2 \max(v_x, 1.0)}$, che filtra la quota $z_{\text{micro}}$ e la velocità $\dot{z}_{\text{road}}$ tramite la cedevolezza radiale del battistrada.
  - Smorzamento viscoelastico non-lineare con saturazione morbida $v_{\text{damp}} = v_{\text{limit}} \tanh((\dot{z}_{\text{road}} - v_z) / v_{\text{limit}})$ con $v_{\text{limit}} = 0.35\text{ m/s}$, che preserva il 100% dello smorzamento nominale del corpo vettura eliminando shock impulsivi e oscillazioni violente di $F_z$ sul fondo rugoso.

---

## 24. Banco Dinamico 7-Post Shaker Rig (`mdo_seven_post_rig.py`, `ui_seven_post_rig.py`)

Il modulo **7-Post Shaker Rig** costituisce lo strumento integrato della suite per la caratterizzazione, taratura e ottimizzazione delle sospensioni e della piattaforma aerodinamica su banco dinamico virtuale a 7 attuatori idraulici.

### 24.1 Architettura di Attuazione & Geometria del Banco
Il banco agisce sul veicolo tramite 7 gradi di forzamento controllati:
1. **4 Attuatori Stradali (Ruota $FL, FR, RL, RR$):**
   - Vincolati al basamento sismico del banco, traslano verticalmente sotto le impronte a terra dei 4 pneumatici imponendo gli spostamenti $z_{\text{post},i}(t)$ e le velocità $\dot{z}_{\text{post},i}(t)$.
   - Simulano eccitazioni sinusoidali mono- o multi-frequenza, sweep in frequenza logaritmici (chirp), profili di pista reali registrati (cordoli, dossi) o rugosità stocastica secondo normativa **ISO 8608**.
2. **3 Attuatori Telaio / Carichi Aerodinamici (Tripode Superiore/Inferiore):**
   - **Post 5 (Anteriore Centrale):** Collegato alla mezzeria dell'asse anteriore ($x = +a_s, y = 0$), applica il carico verticale aerodinamico e inerziale anteriore.
   - **Post 6 (Posteriore Sinistro):** Collegato all'estremità posteriore sinistra del telaio ($x = -b_s, y = +t_r/2$).
   - **Post 7 (Posteriore Destro):** Collegato all'estremità posteriore destra del telaio ($x = -b_s, y = -t_r/2$).
   - La cinematica a tripode riproduce esattamente qualsiasi terna di sollecitazioni di cassa:
     $$F_{\text{down}} = F_5 + F_6 + F_7, \quad M_{\text{pitch}} = a_s F_5 - b_s (F_6 + F_7), \quad M_{\text{roll}} = \frac{t_r}{2} (F_6 - F_7)$$

### 24.2 Sottosistema Multibody a 14 Stati Dinamici Verticali
Lo stato del sistema è definito dal vettore:
$$\mathbf{x} = \begin{bmatrix} z_s & w_h & \phi & \dot{\phi} & \theta & \dot{\theta} & z_{w,fl} & v_{w,fl} & z_{w,fr} & v_{w,fr} & z_{w,rl} & v_{w,rl} & z_{w,rr} & v_{w,rr} \end{bmatrix}^T$$
- **Massa Sospesa ($m_s$):**
  $$m_s \dot{w}_h = \sum_{i=1}^4 F_{\text{susp},i} - F_{\text{aero,tot}} - m_s g$$
  $$I_x \ddot{\phi} = \frac{t_f}{2}(F_{\text{susp},fl} - F_{\text{susp},fr}) + \frac{t_r}{2}(F_{\text{susp},rl} - F_{\text{susp},rr}) + M_{\text{roll,rig}}$$
  $$I_y \ddot{\theta} = -a_s(F_{\text{susp},fl} + F_{\text{susp},fr}) + b_s(F_{\text{susp},rl} + F_{\text{susp},rr}) + M_{\text{pitch,rig}}$$
- **Masse Non Sospese (Mozzi Ruota $m_{u,i}$):**
  $$m_{u,i} \dot{v}_{w,i} = F_{z,\text{tire},i} - F_{\text{susp},i} - m_{u,i} g, \quad \dot{z}_{w,i} = v_{w,i}$$
- **Contatto Pneumatico Monolaterale & Lift-Off:**
  $$\Delta z_i = \delta_{0,i} - (z_{w,i} - z_{\text{post},i}(t))$$
  $$F_{z,\text{tire},i} = \max\left(0,\, K_{tz,i}(P) \Delta z_i + C_{tz}(\dot{z}_{\text{post},i} - v_{w,i})\right)$$
- **Sospensione Completa:**
  $$F_{\text{susp},i} = F_{\text{susp},0,i} + F_{\text{spring},i}(z_{\text{jounce},i}) + F_{\text{damper},i}(v_{\text{jounce},i}) + F_{\text{arb},i}(\Delta z_{\text{axle}})$$
  con modellazione non-lineare Öhlins TTX25 MkII a 4 vie (LSC, HSC, LSR, HSR) o fallback lineare.

### 24.3 Metriche Ingegneristiche, Post-Processing Avanzato & Scorecard
- **Indice di Grip Meccanico (Contact Patch Load Variation - CPLV Multibanda):**
  $$\text{CPLV}_i = \frac{\sigma(F_{z,i})}{\bar{F}_{z,i}} = \frac{\sqrt{\frac{1}{T}\int_0^T (F_{z,i}(t) - \bar{F}_{z,i})^2 dt}}{\bar{F}_{z,i}}$$
  - **Decomposizione Multibanda:**
    - *Low-Band ($0.5 - 4.5\text{ Hz}$):* Oscillazioni del corpo vettura (heave/bounce e pitch della massa sospesa).
    - *High-Band ($8.0 - 18.0\text{ Hz}$):* Oscillazioni di risonanza delle masse non sospese (wheel hop).
  - **Stima Perdita di Aderenza Meccanica (%):** Per effetto della degressività di aderenza con il carico normale, la fluttuazione del carico riduce il grip medio fruibile:
    $$\Delta\mu_{\text{loss}} \approx k_{\text{deg}} \cdot \text{CPLV}^2$$
  - **Monitor Lift-Off:** Conteggio e percentuale di tempo in cui $F_{z,i} \le 1.0\text{ N}$ (distacco completo del pneumatico dal piatto dell'attuatore).

- **Diagnostica Spettrale (FRF Bode, Fase & Coerenza):**
  - Magnitudo trasmissibilità sia in unità lineari $[-]$ che in Decibel $[\text{dB}] = 20\log_{10}(|H(f)|)$.
  - Spettro di fase $\phi(f) \in [-180^\circ, +180^\circ]$ tra eccitazione stradale e risposta telaio.
  - Funzione di coerenza spettrale $\gamma^2(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)} \in [0, 1]$ con soglia minima di linearità $\gamma^2 \ge 0.8$.
  - Riconoscimento automatico e annotazione grafica dei picchi di risonanza (Chassis Heave, Chassis Pitch, Front/Rear Wheel Hop).

- **Diagnostica Smorzatori TTX25 & Bilancio Energetico:**
  - **Cicli di Isteresi Lissajous:** Forza di smorzamento vs velocità stelo ($F_{\text{damper}}$ vs $v_{\text{damper}}$) per asse anteriore e posteriore.
  - **Istogramma Bilstrom a 5 Bande di Velocità:**
    1. *Ultra-Low:* $< 10\text{ mm/s}$ (micro-movimenti di stabilizzazione)
    2. *Low-Speed:* $10 - 25\text{ mm/s}$ (rollio e beccheggio transitorio)
    3. *Mid-Speed:* $25 - 50\text{ mm/s}$ (ondulazioni di pista)
    4. *Mid-High:* $50 - 100\text{ mm/s}$ (cordoli e cambi di pendenza)
    5. *High-Speed:* $> 100\text{ mm/s}$ (impatti severi e dossi)
  - **Bilancio di Potenza ed Energia:**
    - Potenza dissipata istantanea $P(t) = F_d(t) \cdot v_d(t)$ [W] (valori RMS e picco).
    - Energia totale assorbita per ammortizzatore $E_{\text{diss}} = \int P(t) dt$ [J].
    - Rapporto di asimmetria lavoro Compressione / Estensione: $R_{\text{asym}} = \frac{E_{\text{comp}}}{E_{\text{reb}}}$.
    - Monitoraggio contatti tamponi finecorsa (Bumpstop strikes).

- **Controllo Piattaforma Aerodinamica & Stabilità Telaio:**
  - Fluttuazione ride height anteriore e posteriore: RMS ed escursione picco-picco ($\Delta h_{f,\text{pk-pk}}, \Delta h_{r,\text{pk-pk}}$).
  - Variazione dinamica angoli di beccheggio ($\theta_{\text{pk-pk}}$) e rollio ($\phi_{\text{pk-pk}}$).
  - Spettri di densità di potenza (PSD) delle accelerazioni cassa ($\ddot{z}_s, \ddot{\theta}, \ddot{\phi}$).

- **Scorecard Sintetica di Setup (0 - 100):**
  - $\text{Score}_{\text{grip}} = \text{clip}\left(100 \cdot (1 - 2.2 \cdot \text{CPLV}_{\text{avg}}),\, 0,\, 100\right)$
  - $\text{Score}_{\text{aero}} = \text{clip}\left(100 \cdot \exp\left(-\left(\frac{\text{RMS}(\ddot{z}_s)}{1.8} + \frac{\theta_{\text{pk-pk}}}{1.2}\right)\right),\, 0,\, 100\right)$
  - $\text{Score}_{\text{overall}} = 0.55 \cdot \text{Score}_{\text{grip}} + 0.45 \cdot \text{Score}_{\text{aero}}$

### 24.4 Struttura Interfaccia Utente & Visualizzatore a 8 Sub-Tab
L'interfaccia grafica dedicata (`SevenPostRigTab` in `ui_seven_post_rig.py`) è integrata in Workspace 2 di `main_gui_pro.py` e in `main_gui_classic.py`:
- **Pannello Impostazioni Selezionabili (Sinistra):**
  - *Modo di Prova:* Chirp Logaritmico, Chirp Lineare, Sine Dwell (Risonanza a frequenza fissa), Gradino & Ostacoli (Cordolo, Buca, Dosso sinusoidale, Gradino con ritardo interasse $\tau = L/V$), Rugosità ISO 8608 (Classi A-D).
  - *Fasatura:* Heave [++++], Pitch [++--], Roll [+-+-], Warp [+- -+], Ruota Singola (FL, FR, RL, RR).
  - *Attuatori Telaio:* Modo dinamico ($V^2$), Carico costante (N), Disattivato; velocità virtuale e ripartizione anteriore downforce.
  - *Setup Overrides:* Rigidità molle $K_z$ [N/mm], barre antirollio $K_\phi$ [Nm/rad], luce tamponi bumpstop [mm], taratura TTX25 a 4 vie (LSC, HSC, LSR, HSR) con sincronizzazione rapida da veicolo.
  - *Opzioni Post-Processing:* Toggle Magnitudo dB / Lineare, Scala frequenze Log / Lineare, Decomposizione CPLV multibanda / spettro continuo, Mostra soglie allarmi (lift-off, target CPLV 15%).
- **Visualizzatore Analitico Avanzato a 8 Sub-Tab (Destra):**
  1. *Spettro FRF & Coerenza:* Bode completo a 3 subplot sincronizzati (Magnitudo dB/lin con marker risonanze, Fase [deg], Coerenza $\gamma^2$).
  2. *Grip & CPLV Multibanda:* Bar chart 4 ruote + media (Overall vs Low-Band vs High-Band), stima perdita aderenza %, spettro continuo.
  3. *Telemetria 7 Attuatori:* Corsa attuatori stradali [mm], carichi normali $F_z$ [N] con soglia lift-off, forze attuatori aero telaio [N].
  4. *Diagnostica TTX25 & Bilstrom:* Cicli Lissajous $F-v$, istogramma a 5 bande Bilstrom, box metriche energetiche e potenze.
  5. *Piattaforma Aerodinamica:* Tracce ride height temporali, angoli beccheggio/rollio, densità spettrale accelerazioni (PSD).
  6. *Modi Propri & Smorzamento:* Tabella modi propri linearizzati e diagramma dei poli nel piano complesso s ($\sigma$ vs $j\omega$) con linee a $\zeta$ costante.
  7. *Vista Banco 2D (Cinematica):* Canvas cinematico 2D animato con visuale laterale e trasversale, ruote con colorazione verde/rosso per contatto o lift-off, timeline interattiva a 30 FPS.
  8. *Report Tecnico Ingegneristico:* Riepilogo testuale formattato con pulsanti "Copia negli appunti" ed esportazione su file (.txt / .md).

---

## 45. Simulatore 14-DOF: Posizionamento Staging Dietro la Linea, Estensione Tracciati Aperti & Linearità Sterzo Base 2.0

### 45.1 Posizionamento di Partenza (Staging 2m Dietro lo Start)
Nelle precedenti release del simulatore, l'inizializzazione del veicolo (`reset_state`) posizionava la monoposto 10 metri oltre la fotocellula di partenza ($s_0 = +10.0\text{ m}$), impedendo il conteggio del primo giro finché l'auto non compiva un'intera tornata.
L'architettura aggiornata implementa il posizionamento realistico da griglia di partenza / staging:
- **Circuiti Chiusi (`is_closed = True`):**
  L'auto viene posizionata esattamente 2.0 metri dietro la linea di traguardo/partenza lungo la coordinata curvilinea:
  $$s_0 = (L_{\text{total}} - 2.0\text{ m}) \pmod{L_{\text{total}}}$$
  L'orientamento $\psi_0$ è allineato tangenzialmente al tracciato $\hat{t}(s_0)$, puntando in avanti verso il traguardo.
- **Circuiti Aperti (`is_closed = False`):**
  L'auto viene posizionata 2.0 metri prima della fotocellula di partenza:
  $$s_0 = -2.0\text{ m}$$
- **Stato Fotocellula & Lap Timer:**
  La distanza proiettata firmata rispetto al piano del cancello fotocellula:
  $$s_{\text{fwd}} = (x_{\text{car}} - x_{\text{sf}}) \hat{t}_x + (y_{\text{car}} - y_{\text{sf}}) \hat{t}_y \approx -2.0\text{ m}$$
  All'inizializzazione:
  - `_sf_dot_prev = s_fwd` ($\approx -2.0\text{ m}$)
  - `_sf_armed = False`
  - `lap_count = 0` (fase di griglia / staging)
  Al superamento in avanti della linea ($s_{\text{fwd}} \ge 0$), il cancello fotocellula scatta immediatamente: `_sf_armed = True`, `lap_count = 1`, `lap_time_current = 0.0`, attivando la telemetria di Lap 1 dal momento esatto del passaggio sull'asfalto di partenza.

### 45.2 Estensione Automatica Tracciati Aperti (`extend_open_track_start`)
Nei circuiti aperti (es. Acceleration FSAE, Sprint, Hillclimb), il tracciato nativo iniziava a $s = 0$. Il posizionamento dell'auto a $s = -2.0\text{ m}$ avrebbe collocato le ruote nel vuoto (assenza di mesh stradale e grafica).
La funzione specializzata `extend_open_track_start(track_data, ext_len=20.0)` in `track_manager.py`:
1. Calcola il versore tangente d'ingresso $\hat{t}_0 = (\Delta x_0 / L_0, \Delta y_0 / L_0)$.
2. Genera $N_{\text{ext}}$ stazioni all'indietro lungo $-\hat{t}_0$ per un'estensione di $20.0\text{ m}$:
   $$s_{\text{ext}} = [-N_{\text{ext}} \cdot ds,\, \dots,\, -ds], \qquad \mathbf{r}_{\text{ext}} = \mathbf{r}_0 + \hat{t}_0 \cdot s_{\text{ext}}$$
3. Concatena simmetricamente le coordinate di mezzeria, bordi sinistro e destro ($x_l, y_l, x_r, y_r$), quote altimetriche ($z_c, z_l, z_r$) e banking.
4. Identifica l'indice esatto della linea di partenza originale:
   $$\text{sf\_index} = N_{\text{ext}}$$
   dove $s = 0.0\text{ m}$, mentre le stazioni $s \in [-20.0, 0.0)$ costituiscono la corsia di staging.
5. Rigenera le fasce d'erba `extrude_verge_geometry` e la mesh triangolare di collisione 3D `TrackCollisionMesh`:
   - A $s = -2.0\text{ m}$, il raycasting ruota rileva asfalto nominale (surface type 0), normale verticale $\hat{n} = [0, 0, 1]$ e quota corretta.
   - Il renderer OpenGL (`ui/sim_viewport.py`) traccia la linea a scacchi esattamente all'indice `sf_index` ($s = 0.0$), disegnando l'asfalto continuo e la segnaletica orizzontale dietro di essa.

### 45.3 Linearità Sterzo Base a 2.0
Per offrire un controllo ottimale sia su controller gamepad (levette analogiche) che su tastiera e periferiche DirectInput/XInput, la sensitività/linearità dello sterzo è stata portata al valore nominale di riferimento **2.0**:
- **Formulazione dell'Input Shaping:**
  $$\delta_{\text{shaped}} = \text{sign}(\delta_{\text{norm}}) \cdot |\delta_{\text{norm}}|^{\gamma}, \qquad \gamma = 2.0$$
  con deadzone centratura al 4%.
- Un esponente $\gamma = 2.0$ (risposta quadratica progressiva) garantisce massima dolcezza e precisione micrometrica nei piccoli angoli intorno allo zero (controllo del rettilineo e correzioni fini ad alta velocità), conservando allo stesso tempo la piena escursione a fondo corsa per tornanti e manovre d'emergenza.
- Valore integrato come standard predefinito in `core/gamepad_manager.py` (`GamepadManager.__init__`), nel preset di default `Bilanciato (Consigliato)` di `FFB_PRESETS` e nel selettore a doppio spinbox `self.sb_sens` in `ui/sim_main_window.py`.

---

*Ultimo Aggiornamento: Ottobre 2026 — Dynamic Target Suite Version 4.9.1 (14-DOF S/F Staging Behind Start Line, Open Track Backward Extrusion & Steering Linearity 2.0)*












