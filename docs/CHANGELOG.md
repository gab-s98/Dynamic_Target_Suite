# Changelog

All notable changes to the **Dynamic Target Suite** project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.10.0] — 2026-10-08

### Added & Refactored (Direct Collocation UI, Telemetry Data Inspector & Simulator Camera Smoothing / Live-Tuning)
- **Modulo Direct Collocation: Disaccoppiamento Canvas & Allineamento UI (`src/ui/ui_direct_collocation.py`)**:
  - *Disaccoppiamento Canvas*: Eliminato l'accoppiamento con il simulatore e le relative dipendenze circolari; introdotta la classe `CollocationTrajectoryCanvas` con 4 sotto-grafici sincroni (Racing Line vs Mezzeria, Profilo di velocità $v(s)$ con confronto quasi-statico vs ottimo IPOPT, Inviluppo Controlli $a_x(s)$ e $\delta(s)$, ed Ellisse d'Aderenza normalizzata).
  - *Telemetry Data Inspector Multi-Canale (`DataInspectorCanvas`)*:
    - Layout multi-grafico liberamente configurabile: 1x1, 2x1, 3x1 e 2x2.
    - Selezione dinamica dei canali con oltre 14 tracce fisiche e canali derivati calcolati al volo ($a_{tot}$, $\Delta v$, $\dot{\psi}$, $P_{long}$).
    - Cursori verticali sincronizzati (*crosshair*) su tutti i subplot al movimento del mouse con readout numerico istantaneo.
    - Asse delle ascisse commutabile in tempo reale tra ascissa curvilinea $s$ [m] e tempo sul giro $t$ [s].
  - *Pannello Configurazione Tracciato (`TrackConfigPanel`)*:
    - Selezione da libreria preset Formula SAE (*FSAE Autocross*, *FSAE Skidpad*, *FSAE Endurance*, *FSAE Acceleration*) o caricamento file custom CSV/JSON.
    - Parametrizzazione interattiva di larghezza pista $w_{track}$, passo di discretizzazione $\Delta s$, condizioni al contorno e canvas di anteprima geometrica isolato.
  - *Design System & Worker Asincrono*: Integrazione nativa con `ui_theme_engine.py` (palette scura, font `Segoe UI`, badge tecnici ASCII), log di convergenza IPOPT in tempo reale e thread asincrono `CollocationWorker(QThread)`.
- **Modulo Simulatore: Camera Smoothing Onboard & Live Vehicle Parameters Inspector**:
  - *Camera Smoothing Continuo in Prima Persona (`COCKPIT`, `NOSE`, `BUMPER`)*:
    - Risolto il tremolio (*jitter*) su tracciati con variazioni altimetriche e salti di pendenza implementando un filtro esponenziale continuo tempo-invariante ($\alpha = 1 - e^{-\lambda \Delta t}$) con costanti temporali dedicate per coordinate planari, quota altimetrica $z$, look-at target e angoli di beccheggio/rollio.
    - Ortonormalizzazione sferica per i vettori di puntamento ed elevazione, eliminando divergenze e gimbal lock in frenata brusca o compressione.
  - *Finestra Esterna Fluttuante: Live Vehicle Parameters Inspector & Tuning (`LiveVehicleParametersInspector`)*:
    - Finestra non bloccante e non modale (`Qt.Window`, aperta con `.show()`), che non arresta né congela il ciclo OpenGL o il thread fisico a 250 Hz.
    - Tab *Ispezione Telemetria Live*: stream a 25 Hz protetto da lock dei parametri di dinamica, carichi normali $F_{z,i}$, corse sospensioni e grip/temperature pneumatici.
    - Tab *Live-Tuning Parametri Veicolo*: modifica istantanea di rigidezza molle, barre antirollio, click ammortizzatori Öhlins TTX25 MkII (LSC, LSR), aerodinamica ($C_z, C_d, S, \text{bal}_{aero}$), pneumatici ($\mu_x, \mu_y$, pressioni $P_f, P_r$), massa pilota/zavorra e ripartizione statica.
    - Supporto a Auto-Apply live, hot-reload thread-safe nel `PhysicsThread14DOF`, ripristino parametri nominali di fabbrica ed esportazione/importazione JSON.
- **Suite di Test & Build**:
  - Creata la suite `tests/test_collocation_ui_and_live_inspector.py` con 6 test esaustivi.
  - Aggiornato `MDO_Target_Suite.spec` e `scripts/ci_pipeline.py`.

## [4.9.0] — 2026-10-06

### Added & Engineered (7-Post Dynamic Shaker Rig & Advanced Multi-Tab Post-Processing Suite)
- **Motore Fisico 7-Post Shaker Rig (`mdo_seven_post_rig.py`)**:
  - Implementato il sottosistema dinamico verticale a 14 stati per la simulazione e l'analisi modale di Formula SAE e prototipi da corsa.
  - **Cinematica dei 7 Attuatori**:
    - 4 attuatori ruota indipendenti sotto i contact patch ($FL, FR, RL, RR$) per profili di spostamento $z_{post,i}(t)$ e velocità verticali.
    - 3 attuatori telaio (tripode: 1 anteriore centrale sull'asse di simmetria $x = +a_s$, 2 posteriori alle estremità della carreggiata $x = -b_s, y = \pm t_r/2$) che replicano deportanza complessiva $F_{down}$, momento di beccheggio $M_{pitch}$ e rollio $M_{roll}$.
  - **Modalità di Prova Selezionabili**:
    - *Frequency Sweep (Chirp Logaritmico / Lineare)* da $0.1\text{ Hz}$ a $40\text{ Hz}$ con limite velocità attuatore $v_{\text{max}}$.
    - *Sine Dwell (Risonanza a Frequenza Fissa)* per analisi di regimi stazionari ed isteresi cicliche.
    - *Gradino / Cordolo / Buca (Step & Obstacle)* con ritardo tra gli assi $\tau = L/V$ per riprodurre impatti realistici con cordoli ($+15\text{ mm}$), buche o dossi sinusoidali.
    - *Rugosità Stradale ISO 8608 (Classi A, B, C, D)* per test di usura ed eccitazione stocastica ad alta frequenza.
    - *Fasature Canoniche*: Heave [++++], Pitch [++--], Roll [+-+-], Warp [+- -+], ed eccitazione singola ruota (FL, FR, RL, RR).
  - **Modello di Contatto Ruota Unilaterale**: Rigidezza radiale $K_{tz}(P)$, smorzamento carcasse $C_{tz}$ e vincolo di non-negatività $F_z \ge 0\text{ N}$ con rilevamento del distacco ruota (*lift-off*).
  - **Sospensioni & Öhlins TTX25 MkII**: Molle lineari, bumpstop a impaccamento solido progressivo, finecorsa in estensione, barre antirollio ARB e ammortizzatori 4-way non-lineari (LSC, HSC, LSR, HSR) con override dinamici e sincronizzazione da veicolo.
  - **Equilibrio Statico 1g Rigoroso**: Calibrazione esatta dei bracci di massa sospesa $a_s, b_s$ e dei precarichi $F_{susp,0,i}$, garantendo derivate accelerative identicamente nulle a terraferma ($z_i = 0\text{ mm}$).
- **Suite di Post-Processing Avanzato a 8 Sub-Tab (`ui_seven_post_rig.py`)**:
  - *Opzioni di Visualizzazione Selezionabili*: Toggle unità FRF Decibel [dB] vs Lineare [-], scala frequenze Logaritmica vs Lineare, vista CPLV multibanda vs spettro continuo, e commutazione soglie di allarme (lift-off, target CPLV).
  - *Monitor KPI a 8 Card*: Score Setup Rig (0-100), Grip Score (0-100), CPLV Medio (%), Lift-Off (%), Accel Heave/Pitch/Roll RMS, e fluttuazione picco-picco Ride Height.
  - *Tab 1: Spettro FRF, Fase & Coerenza*: 3 subplot sincronizzati con identificazione ed etichettatura automatica dei picchi di risonanza telaio/ruote e soglia coerenza lineare $\gamma^2 = 0.8$.
  - *Tab 2: Grip & CPLV Multibanda*: Confronto a barre tra CPLV complessivo, Low-Band ($0.5-4.5\text{ Hz}$, cassa) e High-Band ($8-18\text{ Hz}$, wheel hop), stima percentuale della perdita di aderenza laterale e spettro continuo ruote.
  - *Tab 3: Telemetria 7 Attuatori*: Tracce temporali delle corse attuatori idraulici stradali [mm], carichi normali $F_z$ [N] con linea di soglia zero, e forze attuatori aero telaio [N].
  - *Tab 4: Diagnostica TTX25 & Bilstrom*: Cicli Lissajous $F-v$, istogramma Bilstrom a 5 bande di velocità (<10, 10-25, 25-50, 50-100, >100 mm/s), box metriche di potenza di picco/RMS, energia dissipata e rapporto asimmetria $E_{\text{comp}}/E_{\text{reb}}$.
  - *Tab 5: Piattaforma Aerodinamica*: Fluttuazioni ride height anteriore/posteriore [mm], angoli di beccheggio/rollio dinamici, e densità spettrale delle accelerazioni (PSD).
  - *Tab 6: Modi Propri & Smorzamento*: Tabella autovalori e frequenze naturali linearizzate, e mappa poli/zeri nel piano complesso s con isolinee di smorzamento $\zeta$.
  - *Tab 7: Vista Banco 2D (Cinematica Animata)*: Animazione 30 FPS laterale e trasversale, colorazione dinamica verde/rosso gomma a terra, timeline interattiva di scrub e selettore velocità ($0.25\times, 0.5\times, 1.0\times$).
  - *Tab 8: Report Tecnico Ingegneristico*: Generatore automatico di report markdown/testuale con copia istantanea negli appunti ed esportazione file TXT/MD.
- **Integrazione UI Pro & Classic**:
  - Inserito come sub-tab nel **Workspace 2 ("Assetto & Componenti Sospensioni")** in `main_gui_pro.py` e in `setup_subtabs` di `main_gui_classic.py`.
  - Nuova voce nel menu: `Strumenti -> Banco Dinamico 7-Post Shaker Rig...`.
  - Esecuzione asincrona multi-thread con `SevenPostRigWorker(QThread)` in `gui_workers.py`.
- **Suite di Test Unitari & CI**:
  - Modulo di test unitari dedicato `tests/test_seven_post_rig.py` con 12 test di verifica fisica ed ergonomica (inclusi Sine Dwell, Step Kerb, Bilstrom 5-bin, Report generator e rendering UI).
  - Aggiornamento specfile `MDO_Target_Suite.spec` per compilazione PyInstaller.

## [4.8.6] — 2026-10-05

### Fixed & Engineered (3D Road Mean Plane Kinematics, 0N Load Anomaly Fix & Real-Time Performance Decoupling)
- **Cinematica Piano Medio Stradale 3D & Risoluzione Anomalia Carichi 0N**:
  - Risolta la causa radice della perdita artificiale di carico verticale ($F_z = 0\text{ N}$) da fermo su dossi, salite e banking: l'espressione precedente $(z_{\text{road},i} - z_{\text{road,macro}})$ cancellava la deflessione statica del pneumatico $\delta_{0,i} \approx 8\text{ mm}$ a causa della variazione di quota longitudinale/trasversale ($\pm 40\text{ mm}$).
  - Implementata la decomposizione geometrica sul piano medio stradale locale:
    $$\theta_{\text{road}} = -\frac{z_{\text{road},f} - z_{\text{road},r}}{L}, \quad \phi_{\text{road}} = \frac{(z_{\text{road},fl} + z_{\text{road},rl}) - (z_{\text{road},fr} + z_{\text{road},rr})}{2 t_m}$$
    $$z_{\text{plane},i} = z_{\text{mid}} \mp \theta_{\text{road}} \frac{L}{2} \pm \phi_{\text{road}} \frac{t_m}{2}, \quad \Delta z_{\text{micro},i} = z_{\text{road},i} - z_{\text{plane},i}$$
    $$\delta_i = \delta_{0,i} - z_{\text{chassis},i} + \Delta z_{\text{micro},i}$$
  - Su pendenze uniformi $\Delta z_{\text{micro},i} \equiv 0$, garantendo $F_z > 0\text{ N}$ continuo e carichi verticali fisici corretti su tutte le 4 ruote in qualsiasi assetto 3D.
  - Aggiornati i momenti ripristinanti e destabilizzanti del telaio rispetto agli angoli relativi $\theta_{\text{rel}} = \theta - \theta_{\text{road}}$ e $\phi_{\text{rel}} = \phi - \phi_{\text{road}}$.
- **Ottimizzazione Throughput Loop Fisico (500–1000 Hz) & Rimozione Colli di Bottiglia**:
  - Eliminata la doppia valutazione duplicata del profilo stradale ed eliminata la 5ª chiamata ridondante all'ODE CasADi (`ode()`) per estrarre $a_{x,\text{raw}}$, riciclando direttamente `_k1[0]` calcolato durante l'integrazione RK4.
  - Disaccoppiamento architetturale: il blocco computazionalmente oneroso di telemetria, valutazione Pacejka MF6.1 vettorializzata, coppie FFB, e instanziazione dizionari è stato confinato alla frequenza video (60 Hz). I tick intermedi della fisica eseguono unicamente il sub-stepping RK4 in ~0.8 ms, garantendo 500 Hz stabili e jitter sub-microsecondo.
  - Sostituita la gestione `step_times` (allocazione/pop di liste continue) con media esponenziale progressiva a zero allocazioni dinamiche.
- **Ottimizzazione Prestazioni GUI & Riduzione Drastica Contesa GIL**:
  - In `sim_main_window.py`: mantenuto il rendering OpenGL/QPainter del viewport a 60 FPS continui, mentre l'aggiornamento testuale degli oltre 70 `QLabel`, barre di avanzamento e parsing QSS di `setStyleSheet()` è stato razionalizzato e cadenzato a 15 Hz con caching degli stati grafici.
  - In `sim_viewport.py`: cadenzato il pruning dell'array di strisciate di frenata (`skid_ribbons`) a 2 Hz anziché rieseguire list comprehension ad ogni frame.
  - In `track_manager.py`: algoritmo di raycast verticale `TrackCollisionMesh.raycast_down` riscritto in proiezione scalare baricentrica 2D (accelerazione di 28x, 46 ms vs 1289 ms su 50k raggi).

## [4.8.5] — 2026-09-20

### Added & Engineered (2D Road Surface ISO 8608, 3D Verge Geometry & C DLL Real-Time Road Contact)
- **Superficie Stradale 2D su Griglia (s, d) & Spettro ISO 8608 Classe A**:
  - Implementato il modulo [`track_manager.py`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/track_manager.py) per la sintesi stocastica 2D dell'elevazione stradale $Z_{\text{surf}}(s, d)$ con densità spettrale di potenza ISO 8608 Classe A ($G_d(n_0) = 4 \times 10^{-6}\text{ m}^3$, $w = 2.0$) e deviazione standard calibrata $\sigma \approx 1.41\text{ mm}$.
  - Filtro di coerenza laterale lorentziano con raggio di correlazione $d_{\text{lat}} = 0.20\text{ m}$, garantendo decadimento esponenziale della cross-correlazione tra profili paralleli con separazione laterale $\Delta d > 0.1\text{ m}$.
  - Filtro ad inviluppo spaziale rettangolare (boxcar) calibrato sull'impronta a terra del pneumatico ($L_{\text{patch}} = 0.12\text{ m}$) lungo la direzione longitudinale $s$, prevenendo picchi impulsivi su $\dot{z}_{\text{road}}$.
  - Motore di campionamento bilineare vettoriale `RoadSurface2D` operante a 1000 Hz ($< 2\ \mu\text{s}$ per query) con wrapping periodico su circuiti chiusi ($s = L \to 0$) e clamping laterale oltre la carreggiata.
- **Iniezione Parametrica nel Grafo CasADi C DLL (Zero Fisica Placebo)**:
  - Esteso il vettore parametri $p$ del solutore CasADi da 17 a 25 elementi in `vehicle_14dof.py`, iniettando le quote di elevazione suolo $[z_{\text{road},fl..rr}]$ (indici 17..20) e le velocità verticali del fondo $[\dot{z}_{\text{road},fl..rr}]$ (indici 21..24) direttamente nell'ODE nativa compilata.
  - Ricompilata la DLL C ad alte prestazioni con MinGW GCC (`ode_14dof_63d988c94920a084.dll`) aggiornando la firma di integrità a `'v11_2d_road_contact'`.
  - Retrocompatibilità al 100% garantita tramite zero-padding automatico nel wrapper `ODEFunctionWrapper` per chiamate legacy o test non-stradali.
- **Geometria Terreno 3D (Verge Mesh Skirt & Rimozione Piano Rigido Z=0)**:
  - Eliminato definitivamente il piano orizzontale rigido verde a $Z=0$ dal render 3D OpenGL, che creava tagli anomali attraverso sezioni in salita, dosso o banking del tracciato.
  - Implementata l'estrusione 3D delle fasce perimetrali di banchina ed erba (`extrude_verge_geometry`) lungo i versori binormali $\pm\mathbf{b}(s)$ con larghezza $W_{\text{verge}} = 12.0\text{ m}$ e drenaggio verticale $h_{\text{drop}} = 0.4\text{ m}$, con ombreggiatura verde matte `#2d4a22`.
  - Texture procedurale asfalto 64x64 a micro-rugosità caricata su GPU con coordinate curvilinee ripetute ($u = s \cdot 1.5, v = d \cdot 1.5$).
- **Dashboard Telemetrica HUD & Indicatore Lift-Off**:
  - Aggiornato l'overlay grafico in basso a sinistra in `interactive_driver_14dof.py`: visualizzazione della deflessione effettiva del pneumatico $\Delta z_i$ [mm], quota suolo locale $z_{\text{road},i}$ [mm], e badge ad alto contrasto visivo `LIFT-OFF` in caso di distacco della ruota dal manto stradale ($\Delta z_i \le 0$ o $F_z \le 1.0\text{ N}$).

## [4.8.4] — 2026-09-20

### Fixed & Engineered (14-DOF Speed Sweep Convergence, 3D Track Invalidation & Camera Stability)
- **Risoluzione Crollo 0.0g nello Speed Sweep 14-DOF / MDO**:
  - Implementata risoluzione globale sicura dei percorsi `.tir` in `params_tire.py`, `mdo_global_optimizer.py`, `mdo_target_optimization.py` e `mdo_skidpad.py`: fallback automatico al file nominale Hoosier se il percorso passato da GUI o setup è vuoto (`""`), una directory o corrotto.
  - Inizializzazione cinematica ad assetto variabile per 14-DOF e 7-DOF ($\delta_{\text{kin},0} = \arctan(L/R)$, $v_{y,\text{kin},0} = -b/R \cdot V_x$, $z_{f,0}, z_{r,0}$ calcolati dal carico aerodinamico $F_{z,\text{aero}}(V_x)$).
  - Validazione rigorosa della norma del residuo su entrambi i solutori (`hybr` e Levenberg-Marquardt `lm`): rimossa l'accettazione acritica di punti non convergenti su gradiente piatto.
  - Criterio di arresto controsterzo calibrato su $\Delta\delta < -1.5^\circ$ (anziché $10^{-4}\text{ rad}$), evitando terminazioni premature su micro-oscillazioni numeriche.
  - Fallback analitico fisico aero-potenziato di sicurezza contro crolli artificiali a 0g ($a_y \ge 1.0\text{g}$).
- **Aggiornamento Dinamico Pista 3D & Invalidation Display List OpenGL**:
  - Aggiunto contatore atomico di versione `_version` in `road_profile.py` e cache invalidation (`invalidate_track_display_list()`) in `interactive_driver_14dof.py`.
  - Ricompilazione immediata del display list OpenGL ad ogni modifica di elevazione, dosso o banking da GUI o Track Designer, senza richiedere il riavvio del simulatore.
  - Ricalcolo quota piano erboso: `z_ground = min(z_car, float(np.min(z_cnt_arr))) - 0.25`, impedendo al piano di terra di sommergere salite, banking o tratti rialzati del circuito 3D.
- **Risoluzione Bug Telecamere del Simulatore 14-DOF**:
  - Telecamere Onboard (Cockpit, Nose, Bumper) rigidamente vincolate alla cassa telaio senza filtro EMA in coordinate mondo, azzerando il ritardo visivo che faceva slittare la visuale fuori dall'abitacolo durante le frenate e le curve.
  - Unwrapping angolare a percorso minimo per l'angolo di imbardata $\psi \in [-\pi, \pi]$: eliminato il giro a 360° della telecamera durante l'inversione di marcia o attraversamento del punto di sella.
  - Compensazione della pendenza longitudinale del tracciato (`grade_rd`): la telecamera Chase non punta più verso l'asfalto durante la percorrenza di pendenze positive.
  - Estensione del range degli slider distanza (0–150 m) e altezza (0–100 m), evitando che il cambio di slider sovrascriva i preset ravvicinati.

## [4.8.3] — 2026-09-20

### Fixed & Engineered (6 Multi-Disciplinary DTS Discrepancies Resolution)
- **Task 1 — ARB Direct Lumped Roll Stiffness & Wheel Rate Refactoring**:
  - Disaccoppiamento completo da geometrie cilindriche e moduli di taglio ($G, d_{\text{tube}}, \text{wall}, L_{\text{bar}}$) in `VehicleParams` e form di setup, consentendo architetture custom a coltelli (blade) e bilancieri.
  - Esposizione diretta di rigidezza a rollio $K_{\phi,\text{arb}}$ [N·m/rad] e wheel rate differenziale $K_{w,\text{arb}} = K_{\phi,\text{arb}} / t^2$ [N/m] con conversioni bidirezionali `get_Kw_arb_f/r()` e `set_Kphi_arb_from_Kw()`.
  - Formulazione anti-simmetrica rigorosa delle forze ARB sul telaio ($F_{\text{arb},fl} = +K_{w,\text{arb}}\Delta z_{\text{susp}}$, $F_{\text{arb},fr} = -K_{w,\text{arb}}\Delta z_{\text{susp}}$) con momento ripristinante esatto $M_{x,\text{arb}} = -K_{\phi,\text{arb}}(\phi - \phi_{\text{axle}})$.
- **Task 2 — Risoluzione Crollo $a_y$ nello Speed Sweep 14-DOF ad alte velocità**:
  - Rimossa la discontinuità di segno `ca.if_else(Vyw > 0, ...)` nella funzione di trim CasADi che causava un salto di ~66 N·m di momento di imbardata attorno a $V_{yw} = 0$.
  - Integrazione dello Jacobiano analitico CasADi `jac14` / `jac_14dof` nel solutore `scipy.optimize.root(method='hybr')` per accelerazione quadratica della convergenza.
  - Sostituita la soglia rigida di residuo con tolleranza dinamica proporzionale al carico verticale $F_z(V_x)$: $\text{tol} = \max(15.0, 0.05 \cdot m \cdot a_y)$, garantendo continuità perfetta di aderenza da 10 a 25 m/s ($>1.46\text{g}$ a 14 m/s).
- **Task 3 — Parametrizzazione Fisica Molle & Precarico Sospensione**:
  - Aggiunti in `VehicleParams` i campi espliciti lunghezza libera a riposo $L_0$ (`spring_free_length_f/r_mm`), precarico installato $\Delta L_{\text{pre}}$ (`spring_preload_f/r_mm`) e proprietà calcolata lunghezza a pacco/droop $L_{\text{inst}} = L_0 - \Delta L_{\text{pre}}$.
  - Equazione di forza molla nel 14-DOF: $F_{\text{spring}} = (F_{z,0} - m_u g) + K_{\text{eff}} z_{\text{jounce}}$ con precarico statico esatto.
- **Task 4 — Riposizionamento Telecamere Nose & Bumper & Filtro EMA 8 Hz**:
  - Telecamera Nose avanzata a coordinate locali veicolo $(x=+1.53\text{ m}, y=0, z=+0.28\text{ m})$ e Bumper a $(x=+1.50\text{ m}, y=0, z=+0.08\text{ m})$ all'apice del musetto/ala anteriore, eliminando il clipping della mesh del cockpit.
  - Implementato filtro passa-basso EMA del 1° ordine a $f_c = 8.0\text{ Hz}$ ($\alpha = 1 - e^{-2\pi f_c \Delta t}$) su posizione camera e look-at target, eliminando lo sfarfallio e il chatter ad alta frequenza del telaio (20–25 Hz).
- **Task 5 — Correzione Segno Gravità Banking Stradale & Sincronizzazione Patch 3D**:
  - Segno gravità proiettata conforme a ISO 8855 (+X avanti, +Y sx, +Z alto): banking a destra ($\phi_{\text{bank}} > 0$) genera pendenza verso $-Y$ con accelerazione $a_{y,\text{road\_grav}} = -g \sin(\phi_{\text{bank}})$.
  - Sincronizzazione quota di contatto delle 4 ruote su profilo 3D con offset laterale effettivo $d_i$ rispetto alla mezzeria: $z_{\text{road},i} = z_{\text{centerline}}(s_i) + d_i \sin(\phi_{\text{bank}})$.
- **Task 6 — Ottimizzazione Tattile LT/RT (Haptics 50-60 Hz & Isteresi $\kappa$)**:
  - Rate-limiting a 50–60 Hz ($\Delta t \ge 16\text{ ms}$) e soglia minima di variazione $\Delta\text{intensity} > 0.02$ sulle chiamate XInput/SDL2, eliminando la saturazione del buffer USB HID del thread fisico a 250–500 Hz.
  - Isteresi su slip ratio di bloccaggio ABS / pattinamento TC: attivazione a $\kappa \ge 0.12$, disattivazione a $\kappa \le 0.08$ con tempo minimo di mantenimento $80\text{ ms}$.
  - Diagnosi tecnica documentata sui limiti del protocollo standard XInput a 2 motori e stripping dei pacchetti trigger rumble sul driver Bluetooth standard di Windows.



### Fixed & Engineered (Newton III Action-Reaction Coupling, Anti-Parasite Decoupled Ground Deflection & 14-DOF Lateral Cornering Stability)
- **Risoluzione Divergenza Energetica in Curva (Audit & Ripristino Newton III Multibody)**:
  - Identificata e rimossa la causa radice dell'instabilità energetica (+86 kW parassiti in transitorio): l'erroneo accoppiamento della rigidezza pneumatico $K_{tz}$ sulla corsa relativa sospensione anziché sulla quota suolo-mozzo, che iniettava $+47.470\text{ N/rad}$ per effetto del rollio telaio generando un polo instabile a $+75.7\text{ s}^{-1}$.
  - Ripristinata la cinematica relativa corner-mozzo coerente in terna inerziale ISO:
    $$z_{\text{chassis},i} = z_s - \theta x_{c,i} + \phi y_{c,i}, \quad \dot{z}_{\text{chassis},i} = w - \dot{\theta} x_{c,i} + \dot{\phi} y_{c,i}$$
    $$z_{\text{jounce},i} = z_{w,i} - z_{\text{chassis},i}, \quad v_{\text{jounce},i} = \dot{z}_{w,i} - \dot{z}_{\text{chassis},i}$$
  - Isolamento della deflessione a terra del pneumatico $\delta_{z,\text{tire},i} = \delta_{\text{tire},0,i} - z_{w,i}$ dal rollio/beccheggio cassa, garantendo pura aderenza gravitazionale/aerodinamica senza anelli di feedback spuri.
- **Accoppiamento Azione-Reazione Diretto (Newton III)**:
  - Forze di sospensione valutate direttamente su corsa elastica $z_{\text{jounce},i}$, barre ARB anti-simmetriche e smorzamento 4-vie TTX25 / lineare applicato sulla velocità relativa $v_{\text{jounce},i}$.
  - Reazioni esatte sulle 4 masse non sospese:
    $$m_{u,i} \ddot{z}_{w,i} = F_{z,\text{tire},i} - F_{\text{susp},i} - m_{u,i} g$$
  - Reazioni coerenti sulla cassa per Heave, Rollio e Beccheggio:
    $$m_s \dot{w} = \sum F_{\text{susp},i} - F_{\text{down}} - m_s g + m_s(u q - v p)$$
    $$M_{x,\text{susp}} = -\frac{t_f}{2}(F_{\text{susp},fr} - F_{\text{susp},fl}) - \frac{t_r}{2}(F_{\text{susp},rr} - F_{\text{susp},rl})$$
    $$M_{y,\text{susp}} = -a(F_{\text{susp},fl} + F_{\text{susp},fr}) + b(F_{\text{susp},rl} + F_{\text{susp},rr})$$
  - Garanzia matematica di dissipazione meccanica: $\dot{E}_{\text{mech}} \le P_{\text{in}}$ ($\dot{E} \le -2050\text{ W}$ in coast-down), $\max\text{Re}(\lambda) = -0.0718$, rollio limitato a $0.62^\circ$ durante coast-down a 60 km/h con sine steer $5^\circ$ a 1.5 Hz.
- **Cache Invalidation & Test Hardening**:
  - Aggiornata signature C DLL in `get_model_signature()` a `'v10_1_action_reaction_ttx25'`.
  - Aggiunto test dedicato `test_lateral_stability_cornering` in `tests/test_14dof.py`.

## [4.8.1] — 2026-09-19

### Added & Engineered (Direct Collocation De-conservatism, High-Contrast UI, Unilateral Tire Contact Law & 500-1500 Hz Multi-Rate 14-DOF Stability)
- **Task 1 — Direct Collocation & Racing Line NLP (CasADi + IPOPT)**:
  - De-conservativismo dell'inviluppo di aderenza: grip laterale effettivo esteso al 98%-99%, velocità massima aumentata a $45.0\text{ m/s}$ ($162\text{ km/h}$), bounds $\xi \in [-42^\circ, 42^\circ]$ e $\delta \in [-28^\circ, 28^\circ]$.
  - Regolarizzazione sui tassi di controllo (jerk minimization: $\Delta a_{x}, \Delta \delta$ con peso $10^{-6}$), eliminando la penalizzazione artificiale sull'ampiezza massima di trazione/frenata.
  - Doppio vincolo di trazione motore: limite di coppia ai bassi regimi ($F_{x,\text{trac}} \le \frac{n_{\text{motors}} T_{\max} \tau_{\text{gear}} \eta}{R_w}$) e limite di potenza elettrica di picco ad 80 kW ($F_{x,\text{trac}} \cdot v \le P_{\max}$).
  - UI Dark Theme ad alto contrasto: testi `#f0f6fc` / `#E0E0E0`, assi `#888888`, sincronizzazione completa con `apply_plot_theme`, sostituzione della linea centerline nera invisibile `'k:'` con grigio chiaro ad alta leggibilità `#888888`.
- **Task 2 — Risoluzione Divergenza & Squat 100% Throttle Launch (14-DOF)**:
  - Riformulata la velocità di trasporto della patch di contatto: $V_{\text{transport},i} = \max(\sqrt{V_{xw}^2 + V_{yw}^2}, |\omega_i| R_w)$ con soglia $\epsilon_v = 0.8\text{ m/s}$, prevenendo l'accumulo errato nei tempi di rilassamento in partenze da fermo.
  - Formulazione rigorosa del contatto verticale con legge unilaterale $F_{z,i} = \min(\text{gate}_i \cdot \max(0, K_{tz}\delta_i + C_{tz} v_{z,i}), F_{z,\max})$ senza suzione/trazione verso il suolo e senza clamp rigidi su $\dot{v}_z$ per preservare l'ordine 4 di RK4.
  - Bounding dei polinomi cinematici anti-squat e clamping del fattore di beccheggio `anti_factor` in $[0.20, 1.40]$.
- **Task 3 — Selettore Multi-Rate 500 Hz / 1000 Hz / 1500 Hz**:
  - Supporto a frequenze di integrazione fino a 1500 Hz nel simulatore interattivo real-time, con signature C DLL continua e array contigui `float64` per ctypes.

## [4.8.0] — 2026-09-18

### Added & Engineered (Full 28-State MATLAB/Simulink 14-DOF Architecture, 2nd-Order Unsprung Masses & Emergent Tire Deflection)
- **Evoluzione Modello da 23 a 28 Stati (Standard 14-DOF MATLAB/Simulink)**:
  - Espanso il vettore di stato simbolico da 23 a 28 variabili:
    - Cassa 6-DOF (8 stati): $[v_x, v_y, r, \phi, \dot{\phi}, \theta, \dot{\theta}, w_{\text{heave}}]$ con aggiunta dell'equazione di moto in heave ($w_{\text{heave}} = \dot{z}_{\text{cg}}$) accoppiata a downforce e reazioni sospensione.
    - Masse non sospese al 2° ordine (8 stati): $[z_{fl}, z_{fr}, z_{rl}, z_{rr}]$ e velocità di compressione $[vz_{fl}, vz_{fr}, vz_{rl}, vz_{rr}]$.
    - Wheel spin (4 stati): $[\omega_{fl}, \omega_{fr}, \omega_{rl}, \omega_{rr}]$.
    - Deflessioni elastiche carcassa MF6.1 (8 stati): $[u_{x,fl}, u_{y,fl}, u_{x,fr}, u_{y,fr}, u_{x,rl}, u_{y,rl}, u_{x,rr}, u_{y,rr}]$.
- **Dinamica 2° Ordine Masse Non Sospese & Forze Emergenti**:
  - Superata la formulazione quasi-statica $\tau_z$: il carico normale $F_{z,i}$ emerge fisicamente dalla deflessione radiale della gomma ($K_{tz}$ da file `.tir` e smorzamento $C_{tz}$ in `VehicleParams`).
  - Dinamica dei mozzi derivata dalla 2ª legge di Newton:
    $$m_u \dot{v}z_i = -(K_{tz} + K_{\text{eff},i}) z_i - C_{tz} vz_i - F_{\text{damp},i}$$
    $$\dot{z}_i = v_{\text{susp},i}$$
- **Accoppiamento Cinematico Velocità Relative Sospensione**:
  - Gli ammortizzatori TTX25 (o lineari) e le velocità di corsa sono governati dalle velocità relative tra telaio e mozzo ruota:
    $$v_{\text{susp},i} = vz_i - w_{\text{heave}} \mp \frac{t}{2}\dot{\phi} \pm (a,b)\dot{\theta}$$
  - Assicura smorzamento perfetto in rollio, beccheggio e heave.
- **Risoluzione Asimmetria Camber Thrust in Rettilineo**:
  - Rimosso il clamping monodirezionale spuro $V_{yw} > 0$ che troncava il camber thrust a $V_{yw} = 0$, ripristinando la perfetta simmetria L/R in rettilineo.
- **Simulatore Interattivo & Telemetria a 500 Hz**:
  - Aggiornato `interactive_driver_14dof.py` a 500 Hz con nuovi canali telemetrici per heave e velocità verticali mozzi (`w_heave_ms`, `vz_fl_ms`..`vz_rr_ms`).
- **Verifica & Stabilità**:
  - Eseguiti e superati tutti i 193 test unitari della suite, inclusi smoke test headless GUI e ricompilazione PyInstaller standalone.

## [4.7.4] — 2026-09-17

### Fixed & Engineered (Spinout Energy Dissipation, Combined Slip Clamping, Vmag Relaxation & 14-DOF Kinematic Anti-Features Integration)
- **Risoluzione Roteamento Perpetuo Non Dissipativo in Testacoda**:
  - Risolto il bug di iniezione spuria di energia in derapata: nel modello Pacejka MF6.1 (`tire_model_mf61.py`), la funzione di peso combinato $G_{xa}$ entrava nel secondo quadrante per angoli di deriva $\alpha > 25^\circ$, assumendo valori negativi (fino a $-0.40$) e invertendo il segno delle forze longitudinali (trasformando le resistenze in accelerazione propulsiva). Applicato il clamping non-negativo $G_{xa} = \text{ca.fmax}(G_{xa}, 0.0)$ e $G_{yk} = \text{ca.fmax}(G_{yk}, 0.0)$.
  - Riformulata la velocità di avanzamento della patch di contatto $V_{mag}$ in `vehicle_14dof.py`: sostituita la sola componente longitudinale con la velocità planare totale del mozzo ruota $V_{mag} = \sqrt{V_{xw}^2 + V_{yw}^2 + \epsilon_v^2}$ su tutte e quattro le ruote. Nelle scivolate laterali a $V_{xw} \approx 0$, la frequenza di rilassamento $\frac{V_{mag}}{\sigma_y}$ rimane attiva e fisica, impedendo l'accumulo di deflessioni fittizie multi-metro.
  - Introdotta la saturazione fisica della carcassa (limite di slittamento Coulombiano): le deflessioni elastiche $u_x, u_y$ sono limitate dolcemente a $1.5 \sigma$ tramite $\text{ca.tanh}(u / u_{\lim}) \cdot u_{\lim}$, eliminando l'effetto fionda elastico post-testacoda.
  - Corretta la condizione di quiete e smorzamento a bassa velocità in `interactive_driver_14dof.py`: introdotta la velocità periferica dovuta all'imbardata $v_{\text{motion}} = \sqrt{v_{\text{planar}}^2 + (r \cdot L/2)^2}$. La soppressione numerica delle deflessioni agisce solo quando il veicolo è fermo sia traslazionalmente che rotazionalmente ($v_{\text{motion}} < 0.08\text{ m/s}$), consentendo ai pneumatici di generare la piena coppia frenante dissipativa durante qualsiasi rotazione su se stessi.
  - L'energia cinetica in testacoda decresce ora monotonicamente fino all'arresto completo in circa $1.5\text{ s}$ ($r \to 0$, $E_{\text{kin}} \to 0$).
- **Integrazione Antifeature nel Modello 14-DOF (Anti-Dive, Anti-Squat, Anti-Lift)**:
  - Accoppiati i polinomi cinematici di 3° grado $AD(z_f)$, $AS(z_r)$ e $AL(z_r)$ (derivati dai rilievi Adams Car ET-18) alla dinamica longitudinale e di beccheggio in `vehicle_14dof.py`.
  - Ripartito il trasferimento di carico longitudinale tra quota geometrica (reagita rigidamente dai triangoli e braccetti della sospensione) e quota elastica (che comprime le molle e gli ammortizzatori):
    $$M_{\text{pitch,destab}} = m A_x h_{cg} \cdot [ \text{blend}_{\text{acc}} (1 - AS) + \text{blend}_{\text{brk}} (1 - (AD \cdot BB + AL \cdot (1 - BB))) ]$$
  - Il carico verticale totale a terra sui quattro pneumatici rimane rigorosamente conservato per garantire il pieno grip, mentre l'angolo di beccheggio ($\theta$) e le escursioni ($z_i$) riflettono fedelmente le geometrie anti-affondamento in frenata e anti-squat/pro-squat in accelerazione.
  - Aggiornato l'hashing signature dell'ODE a `'v5_antifeatures_and_spinout_damping'` per la rigenerazione automatica della libreria C compilata ad alta velocità.
- **Suite di Test**:
  - Aggiunto il test unitario `test_12_spinout_dissipation_and_antifeatures` in `test_interactive_driver.py` (12/12 PASSED). Test 14-DOF (7/7 PASSED).

## [4.7.3] — 2026-09-17

### Added & Engineered (Decoupled Front Wheel Inertia, C ODE Cache Invalidation, 9.5% Lateral-Derated TCS, Pitch Load Transfer & 0.97g Acceleration)
- **Risoluzione Critica Cache DLL C compilate CasADi (`ode_14dof_<hash>.dll`)**:
  - Identificato e risolto il fattore radice dello stallo a 0.42g: `vehicle_14dof.py` caricava una libreria binaria C compilata il 16/09/2026 (`ode_14dof_a62b0b2fa443a1cb.dll`) a causa dell'hash di firma invariato (`v3_combined_mirror_toe`). Di conseguenza, le nuove equazioni di trasferimento di carico in beccheggio e la scalatura della pressione non venivano fisicamente eseguite nel loop a 500 Hz.
  - Aggiornato l'hashing signature a `'v4_pitch_load_transfer_and_pressure_decoupled_iw'` includendo esplicitamente `tire_pressure_f_bar`, `tire_pressure_r_bar`, `Iw_f` e `Iw_r`.
  - Rimossi tutti i binari obsoleti e ricompilata la DLL C ad alta velocità `ode_14dof_f26e05fc66b771d2.dll`, allineando la precisione numerica tra simbolico CasADi e C compilato all'ordine di $10^{-16}$.
- **Disaccoppiamento Inerzia Ruote Anteriori/Posteriori ($I_{w,f} = 0.30\text{ kg}\cdot\text{m}^2$ vs $I_{w,r} = 1.20\text{ kg}\cdot\text{m}^2$)**:
  - In un veicolo RWD, l'avantreno non ha motore, trasmissione o semiassi (cerchio 10", slick, disco flottante e mozzo hanno $I_{w,f} \approx 0.30\text{ kg}\cdot\text{m}^2$).
  - La precedente inerzia uniforme $I_w = 1.20\text{ kg}\cdot\text{m}^2$ costringeva le ruote anteriori a generare una resistenza parassita da frenatura rotazionale di $-473\text{ N}$ a $-523\text{ N}$ durante la partenza lanciata, dissipando oltre $-0.16g$ di spinta longitudinale netta.
  - Disaccoppiati i parametri `Iw_f: float = 0.30` e `Iw_r: float = 1.20` in `VehicleParams` e nell'integratore ODE 14-DOF. La forza frenante parassita all'anteriore crolla a meno di $-92\text{ N}$, liberando $+0.15g$ di accelerazione netta.
- **Ricalibrazione Controllo Trazione (TCS) a $\kappa=0.095$ & Derating Laterale Antitrottola**:
  - Il picco di aderenza dello pneumatico `Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir` si colloca a $\kappa = 0.095$ (9.5% di slittamento). Oltre tale soglia, per via dell'ellisse d'attrito (Combined Slip), l'aderenza laterale $F_y$ crolla di oltre il 75%, innescando testacoda e snap oversteer fulminei non appena si dà gas a ruote sterzate.
  - Implementato in `interactive_driver_14dof.py` il **derating laterale dinamico**: $\kappa_{\text{target}} = \max(0.095 - 0.045 \cdot (|\delta| / 15^\circ), 0.04)$. In inserimento e percorrenza di curva, il target di scorrimento si abbassa per preservare fino al 70% di capacità laterale $F_y$ al retrotreno.
  - Rimosso il deadband da fermo (`vx_curr > 0.05`), garantendo intervento istantaneo del TCS fin dal primo millisecondo di partenza lanciata a 0 km/h con modulazione morbida (floor cut $0.15$, gain $6.0$).
  - HUD dinamico integrato nel tachimetro con badge `[TCS: ATTIVO]`, `[TCS: -XX%]` e `[TCS: OFF]`, commutabile via tastiera (`T`) o interfaccia grafica.
- **Trasferimenti di Carico Dinamici in Beccheggio (Pitch Load Transfer)**:
  - Implementato in `vehicle_14dof.py` il trasferimento dinamico di carico in beccheggio $dF_{z,\text{long}} = \frac{K_\theta(-\theta) + C_\theta(-\dot{\theta})}{L_{\text{sym}}}$, dove $K_\theta = 2 K_{z,f} a^2 + 2 K_{z,r} b^2$ e $C_\theta = 2 C_{z,f} a^2 + 2 C_{z,r} b^2$.
  - In accelerazione massima (squat $\theta \approx -0.49^\circ$), il carico verticale sulle ruote posteriori aumenta dinamicamente da $1475\text{ N}$ statici a oltre **$2003\text{ N}$** dinamici, con rigida conservazione della forza normale totale veicolo ($2864.5\text{ N}$).
- **Accoppiamento Pressione di Esercizio Pneumatici Reale (0.85 bar / 12.3 psi)**:
  - Integrata la pressione di esercizio reale (`0.85 bar` vs `18.1 psi` nominale di banco) nelle equazioni di forza longitudinale MF6.1. Con $dp_i = -0.32$, l'espansione dell'impronta a terra incrementa il grip longitudinale $\mu_x$ del $+46.3\%$ tramite `PPX3`, portando il coefficiente d'attrito di picco da $1.00$ a $1.47$ sul file base obbligatorio `_2_3.tir`.
- **Prestazioni di Accelerazione Longitudinale Verificate ($0.90g - 0.97g$ Continuo)**:
  - L'azione combinata di grip a 0.85 bar, trasferimento di carico in squat, disaccoppiamento inerzia $I_{w,f}$ e regolazione TCS a $\kappa=0.095$ eroga un'accelerazione longitudinale continua stabilizzata a **$0.90g - 0.97g$** (da 0 a 45 km/h in meno di 1.4 secondi), eliminando in via definitiva ogni limite a 0.42g.
- **Suite di Test**:
  - Test unitari dedicati `test_interactive_driver.py` (11/11 PASSED), `test_14dof.py` (7/7 PASSED), `test_b_segment_suv.py` (10/10 PASSED). Suite completa 189/189 PASSED.

## [4.7.2] — 2026-09-17

### Fixed & Engineered (Ergonomia Inspector, Rolling Resistance Crr, Dual Power Curves Accel, Virtual Spool & Standstill Force Deadband)
- **Ergonomia Inspector Pro & Sottosezioni Collassabili (`CollapsibleGroupBox`)**:
  - Implementata la classe `CollapsibleGroupBox` con pulsante `[-] / [+]` e supporto al doppio clic per miniaturizzare/espandere qualsiasi gruppo.
  - Convertite le 6 sezioni di `tab_dynamic` in sezioni collassabili (con default compresso per Setup Manuale Dinamico e AutoX/Endurance, liberando oltre 350px di altezza).
  - Racchiusi `tab_dynamic` e `tab_sweep` in aree `QScrollArea` con barre di scorrimento reattive.
  - Sbloccato lo splitter verticale sopra la console di log (`self.inspector_splitter`), consentendo un ridimensionamento fluido verso l'alto e verso il basso senza vincoli minimi rigidi.
- **Coefficiente di Resistenza al Rotolamento ($C_{rr}$)**:
  - Aggiunto `crr: float = 0.015` in `VehicleParams` come Single Source of Truth, con registrazione nel catalogo `MODEL_PARAM_CATALOGUE["ALL"]` sotto *"Aerodinamica (Tutti)"* e limiti `[0.005, 0.035]`.
  - Integrato l'override del parametro `QSY1` in Pacejka MF6.1 (`tire_model_mf61.py`) e nel calcolo di drag del modello 3-DOF (`vehicle_3dof.py`).
- **Doppia Curva di Potenza Elettrica vs Meccanica in Accelerazione**:
  - Chiarita la distinzione fisica tra limite energetico dell'accumulatore FSAE ($P_{\text{elec}} \le 80\text{ kW}$) e potenza meccanica motrice utile alle ruote ($P_{\text{wheel}} \approx 64\text{ kW}$, per via dei rendimenti $\eta_{\text{motor}} \approx 88\%$ e $\eta_{\text{drivetrain}} = 0.95$).
  - Aggiornato il grafico di telemetria dell'evento di accelerazione tracciando contemporaneamente $P_{\text{elec}}$ e $P_{\text{wheel}}$ con scala dedicata.
- **Risoluzione Crash Tab Setup_Vincoli su "Mostra Tutti i Parametri (ALL)"**:
  - Protetto il parser della tabella parametri in `_rebuild_params_panel` contro valori stringa non numerici (`vehicle_type`, `drivetrain_type`, `diff_type`), visualizzandoli come righe descrittive protette senza errori di float formatting.
- **Sincronizzazione Virtual Spool & Regime Motori nel Simulatore**:
  - Risolto il bug di trazione fantasma a zero gas: quando `cmd['throttle'] < 0.01`, la coppia motrice $T_{RL} = T_{RR} = 0.0\text{ Nm}$ a prescindere dall'angolo di sterzo, eliminando qualsiasi propulsione indesiderata in ingresso curva.
  - Per differenziali `spool`, forzata l'uguaglianza rigorosa delle velocità angolari $\omega_{RL} \equiv \omega_{RR} = 0.5(\omega_{RL} + \omega_{RR})$ nello stato dell'ODE ad ogni substep.
  - Sincronizzata la telemetria dei regimi motore ai giri ruota in tempo reale ($\text{RPM} = \omega \cdot i_g \cdot \frac{60}{2\pi}$).
- **Deadband Forze Longitudinali da Fermo ($F_x = 0\text{ N}$)**:
  - Introdotto un filtro deadband per le forze longitudinali da fermo ($v < 0.05\text{ m/s}$ con gas e freno rilasciati), eliminando l'offset numerico di banco prova Pacejka ($S_{Vx}$) e la proiezione statica di convergenza ($-28\text{ N}$ ant, $-27\text{ N}$ post), azzerando esattamente $F_x$ a veicolo fermo.

## [4.7.1] — 2026-09-16

### Fixed & Engineered (SSOT Vehicle Mass 292 kg, 1D Sweep Roll Stiffness Coupling, PRO Transient Sensitivity Tab & Simulator Full Dual-Motor Torque)
- **Single Source of Truth (SSOT) Massa Veicolo (292.0 kg Totale con Pilota)**:
  - Standardizzata la massa base di default a 292.0 kg (`VehicleParams.m = 292.0`), rappresentando in modo coerente e univoco l'auto pronta corsa con pilota a bordo.
  - Sincronizzati i file di setup JSON (`et18_2026_kinematics.json`, `default_vehicle.json`, `et17_2025_kinematics.json`).
  - Rimossa l'addizione ridondante `+ m_driver` nel simulatore 14-DOF (`interactive_driver_14dof.py`), uniformando i badge del peso a 292.0 kg.
- **Accoppiamento Fisico Rigidezze di Rollio & LLTD negli Sweep 1D**:
  - In `mdo_sensitivity.py` e `transient_sensitivity.py`, la variazione parametrica di $K_{\phi,f}$ o $K_{\phi,r}$ ricalcola automaticamente $LLTD_{\text{elastic}} = K_{\phi,f} / (K_{\phi,f} + K_{\phi,r})$.
  - Aggiunti i parametri indipendenti `lltd` (a rigidezza totale costante) e `roll_stiffness_tot` (a ripartizione costante).
- **Integrazione Sensibilità Transitoria & Centri di Rollio nel Layout PRO**:
  - Risolto il conflitto di denominazione di `self.tab_sensitivity` in `main_gui_classic.py` rinominando il modulo in `self.tab_transient_sensitivity`.
  - Integrato il modulo in `main_gui_pro.py` (Workspace 3) con etichetta pulita `"Sensibilità Transitoria & RC"` (rimossa la dicitura legacy `(Tip #28, #30)`).
  - Allineato il selettore di navigazione reattiva `_navigate_to_widget`.
- **Sblocco Erogazione Coppia Dual Motor nel Simulatore (100 Nm / 800 Nm)**:
  - Eliminato il blocco di taglio coppia empirico (TCS choke hardcoded su $\kappa > 0.10$) in `interactive_driver_14dof.py` che tagliava arbitrariamente la coppia a 50 Nm e l'accelerazione longitudinale a 0.46 g.
  - Il propulsore eroga ora la coppia piena di 100 Nm per motore (380-400 Nm per ruota, ~760-800 Nm all'asse posteriore), affidandosi alla gestione fisica del modello propulsivo EMRAX 188HV / 80 kW.
  - Corretta la stringa descrittiva nel pannello HUD in `(Coppia Ruota = 4 × Motore)`.
- **Allineamento Test Suite**:
  - Aggiornati i test di sensibilità, gradiente di sottosterzo ed eventi di accelerazione alla massa nominale di 292.0 kg: **189/189 test passati con successo**.

## [4.7.0] — 2026-09-16

### Added & Engineered (High-Fidelity EMRAX 188HV LC Electric Powertrain, 4:1 Gearbox Reduction, 2D Bivariate Efficiency Map & 80 kW FSAE Accumulator Energy Meter Limit)
- **Modello Propulsivo Elettrico ad Alta Fedeltà (`powertrain_electric.py`)**:
  - Implementata la classe `EMRAX188HV` calibrata rigorosamente sulle curve del costruttore (`motor_curve/01_power_torque.png` e `02_efficiency_speed.png`) a **Peak Power** (~60 kW a 6200-6500 RPM) e **Peak Torque** (plateau a 100 Nm da 500 a 5500 RPM, 70 Nm a 8000 RPM).
  - Approssimazione simbolica differenziabile continua `casadi_peak_torque` con operatore soft-min calibrato ($k=0.08$) e cutoff sigmoideo per solutori NLP (IPOPT/MUMPS) e Direct Collocation.
- **Cinematica di Riduzione 4:1 Dual-Motor**:
  - Architettura a due motori posteriori indipendenti legati alle rispettive ruote con rapporto di riduzione fisso $i = 4.0$ ($\omega_{motor} = 4.0 \cdot \omega_{wheel}$).
  - Moltiplicazione di coppia con rendimento di trasmissione $\eta = 0.95$, erogando fino a $380\text{ Nm}$ per ruota ($760\text{ Nm}$ all'asse posteriore).
- **Mappa di Rendimento 2D $\eta(\text{RPM}, T)$**:
  - Interpolatore bilineare `RegularGridInterpolator` $17 \times 11$ con sweet spot island $\ge 96\%$ (picco $96.6\%$ a 3800 RPM e 44 Nm, estesa tra 2300-5700 RPM e 30-57 Nm) e contorni al 95%, 94%, 90-94%, 86-90%.
- **Limite Elettrico Accumulatore a 80 kW (Regolamento Ufficiale FSAE EV Energy Meter)**:
  - Il vincolo di potenza massima di 80 kW è applicato alla **potenza elettrica totale prelevata dall'accumulatore** ($P_{elec,RL} + P_{elec,RR} \le 80.0\text{ kW}$).
  - La funzione `allocate_dual_motor_torques` calcola $P_{elec,i} = P_{mech,i} / \eta_i$ e, in caso di saturazione dell'accumulatore, ripartisce gli 80 kW disponibili **preservando il differenziale di momento d'imbardata $\Delta P = P_R - P_L$ richiesto dal Torque Vectoring** (supportando ad esempio distribuzioni asimmetriche 50 kW esterna / 30 kW interna).
- **Integrazione Completa Nei Modelli & Simulatori**:
  - `vehicle_3dof.py`: Inviluppo di trazione $F_{x,\max}(v_x)$, punto di lavoro e rendimento istantaneo.
  - `vehicle_7dof.py` & `vehicle_14dof.py`: Metodo `apply_powertrain_limits` sulle velocità di spin delle ruote posteriori.
  - `interactive_driver_14dof.py`: Integrazione nel ciclo fisico a 500 Hz con gruppo HUD `pt_group` (barra di carico batteria 0-80 kW con colore dinamico verde/giallo/arancione/rosso, potenza meccanica, RPM RL/RR e rendimento).
  - `mdo_acceleration.py`: Simulazione sprint 75m con consumo energetico in Wh (elettrico e meccanico) e rendimento medio $\eta_{avg}$ (~76.8%).
  - `mdo_dynamic_events.py`: Rispetto dei limiti motopropulsori in Step Steer, Slalom, DLC ed Endurance.
- **Suite di Test & Build PyInstaller**:
  - Creata la suite `tests/test_powertrain_electric.py` con 15 test unitari dedicati.
  - Test suite complessiva verificata: **189/189 test passati con successo**.
  - Aggiornato `MDO_Target_Suite.spec` e ricompilato l'eseguibile standalone `dist/MDO_Target_Suite/MDO_Target_Suite.exe` verificato in esecuzione headless.

## [4.5.5] — 2026-09-14

### Added & Engineered (Dual-Motor Acceleration Physics, ARB Digitized Curves, Resizable Solver Console, ET-18 Bounds Alignment & Stability Aperiodic Fixes)
- **Fisica Accelerazione Quasi-Statica Dual Motor RWD (ET-18 Calibration)**:
  - Calibrated dual electric motor powertrain architecture in `mdo_acceleration.py` and `interactive_driver_14dof.py`.
  - In dual motor configuration with reduction ratio $4.0:1$, each individual rear wheel receives up to $100\text{ Nm} \times 4.0 = 400\text{ Nm}$ ($800\text{ Nm}$ / $3333\text{ N}$ maximum total tractive torque at the rear axle, constrained by $80\text{ kW}$ battery power and longitudinal Hoosier grip $\mu_x \approx 1.62$).
  - Corrected wheel torque distribution logic in `interactive_driver_14dof.py` lines 1082 and 1283.
  - Reduced 75m sprint acceleration time from sluggish $4.75\text{ s}$ ($a_x = 0.74\text{ g}$) to realistic race performance **$3.92\text{ s}$** ($a_{x,\max} = 1.16-1.25\text{ g}$, $v_{\text{final}} = 101.4\text{ km/h}$).
- **Console Output Solutore Ridimensionabile & Finestra Dedicata**:
  - Replaced static layout with fluid `QSplitter(QtCore.Qt.Vertical)` in `main_gui_pro.py`, removing fixed 110px limit and allowing drag-to-expand.
  - Implemented `ConsolePopoutDialog` in `gui_dialogs.py` sharing `source_text_edit.document()` with live bi-directional sync, line counter, clear, copy/search, and non-modal technical dark styling.
  - Added dedicated button "Finestra Dedicata" to undock and inspect long optimization logs.
- **Risoluzione Frequenze Naturali (0 Hz) & Regimi Aperiodici Instabili**:
  - In `mdo_global_optimizer.py` and `main_gui_classic.py`, detected when $\det(A) \le 0$ ($v_x > v_{\text{crit}}$) where state-space lateral-yaw matrix has real eigenvalues with opposite signs ($\lambda_1 > 0$, divergent yaw).
  - Eliminated misleading $0.00\text{ Hz}$ oscillatory frequency artifact, returning `f_n_hz = None`, `omega_n_rads = None`, `zeta = None`, and displaying `"Aperiodico (Instabile)"` / `"N/A"`.
- **Allineamento Range di Ottimizzazione ET-18 & Penalità Geometrica Intelligente**:
  - Realigned default initial guess and parameter bounds in `mdo_global_optimizer.py` and `main_gui_classic.py` with `setups/et18_2026_kinematics.json`:
    $L \in [1.525, 1.550]\text{ m}$, $t_f \in [1.220, 1.280]\text{ m}$, $t_r \in [1.110, 1.160]\text{ m}$, $K_\phi \in [15000, 60000]\text{ Nm/rad}$, $K_z \in [20000, 60000]\text{ N/m}$.
  - Disabled modification of monocoque dimensions ($L, t_f, t_r$) by default in the optimization GUI.
  - Replaced unconstrained track widening penalty with normalized quadratic deviation penalty from nominal monocoque dimensions:
    $$\text{geom\_penalty} = \text{penalty\_factor} \times 5.0 \times \left[ \left(\frac{L - L_{nom}}{L_{nom}}\right)^2 + \left(\frac{t_f - t_{fnom}}{t_{fnom}}\right)^2 + \left(\frac{t_r - t_{rnom}}{t_{rnom}}\right)^2 \right]$$
- **Calcolo Vertici Rombo ARB da Curve Digitalizzate**:
  - Implemented `ARBDiamondConfig.from_digitized_curve_05_ideal()` and `from_digitized_curve_06_real()` in `mdo_arb_diamond_balance.py`.
  - Generated and saved presets `05_Ideale_Telaio_Rigido.json` and `06_Reale_Torsione_Inclusa.json` in `setups/arb_diamond/`.
  - Added dedicated preset loader buttons ("Curva 05: Ideale" e "Curva 06: Reale") in Tab Rombo ARB of `main_gui_classic.py`.
- **Correzione Errore Sweep 1D di Sensitività & Documentazione Punteggi**:
  - Fixed `NameError: name 're' is not defined` in `main_gui_classic.py:8629`.
  - Verified FS-Rules 2026 Section D 9.1.1 quadratic scoring calculation.
- **Suite di Test Unitari**:
  - Created `tests/test_acceleration_and_diamond.py` with 6 dedicated test cases.
  - Full test suite verified: **167/167 tests passing**.

## [4.5.4] — 2026-09-14

### Added & Engineered (Balanced Neutral Target MDO, Clean Cornering Stiffness, Peak Stability Frontier & 14-DOF Roll Coupling)
- **Assetto Neutro Bilanciato (`target_neutral` Objective Mode)**:
  - Added dedicated objective mode in `mdo_global_optimizer.py` and GUI (`Assetto Neutro Bilanciato (Max Ay + Target Kus)`):
    $$J(X) = -a_y(X) + w_{us} \cdot 0.25 (K_{us}(X) - K_{us,\text{target}})^2 + \text{penalty}_{\text{geom}}(X)$$
  - Integrated customizable target understeer input (`Target K_us: 0.00 deg/g`) in Single Run and Multi-Speed Sweep settings.
- **Rigorizzazione Fisica del Gradiente di Sottosterzo ($K_{us}$) ed Eliminazione del Camber Thrust**:
  - Isolated pure cornering stiffness $\partial F_y / \partial \alpha$ by evaluating lateral force at $\gamma = 0$ and $d_\alpha = 1.0^\circ$, eliminating $-490\text{ N}$ of static camber thrust and conicity/ply-steer offset contamination ($S_{Vy0}$).
  - Restored true baseline physical understeer gradient ($K_{us} \approx -0.11^\circ/\text{g}$) for the nominal FSAE prototype.
- **Accoppiamento Rigidezze di Rollio 14-DOF ($K_{\phi,f}, K_{\phi,r}$)**:
  - Connected suspension and anti-roll bar stiffnesses to the dynamic load transfer distribution:
    $$LLTD_{\text{elastic}} = \frac{K_{\phi,f}}{K_{\phi,f} + K_{\phi,r}}$$
    ensuring that 14-DOF front/rear ARB adjustments directly modulate dynamic slip angle saturation and handling balance.
- **Condizione di Arresto al Picco Fisico di Sterzata ($\partial a_y / \partial\delta = 0$)**:
  - Implemented early peak detection during limit $a_y$ solving (`delta_k < delta_{k-1} - 1e-4 rad` past $0.5\text{g}$), terminating search at the true physical cornering limit before the unstable counter-steering branch.
  - Eliminated the numerical artifact where the optimizer favored dangerous, violently oversteering equilibria beyond the tire saturation limit.
- **Frontiera di Stabilità nell'Handling Diagram di Pacejka**:
  - Preserved the full arc-length continuation method to display the descending post-peak branch.
  - Added interactive toggle and red dash-dotted envelope (**Frontiera di Stabilità**) interconnecting the peak steer saturation points across all evaluated speeds.
- **Calibrazione Penalità SLSQP**:
  - Calibrated understeer deficit penalty to prevent severe boundary pinning ($wd_f = 65\%, LLTD = 65\%$).
- **Test & Automazione CI**:
  - Added dedicated unit test suite `tests/test_mdo_neutral_target.py` (5 unit tests covering pure cornering stiffness, 14-DOF roll stiffness coupling, physical peak detection, and target neutral convergence).
  - Integrated into `scripts/ci_pipeline.py` Stages 1 & 2; 161 unit tests and headless GUI smoke tests passing.

## [4.5.3] — 2026-09-13

### Changed & Unified (ET-18 Default Prototype Across Suite & Dynamic Tire Native ODE)
- **ET-18 Prototype Single Source of Truth**:
  - Configured `VehicleParams` dataclass defaults in `params_vehicle.py` to match the ET-18 prototype: $m = 230.0\text{ kg}$, $h_{cg} = 0.28\text{ m}$, $wd_f = 0.485$, $lltd = 0.525$, $\text{caster}_f = 3.0^\circ$, $\text{kpi}_f = 8.2^\circ$, $\text{scrub\_radius}_f = 0.0368\text{ m}$, $\text{mechanical\_trail}_f = 0.012\text{ m}$, and 2026 Adams Car kinematic polynomial maps.
  - Updated `VehicleParams.nominal_fsae_params()` to load `setups/et18_2026_kinematics.json` by default.
  - Synchronized `setups/default_vehicle.json` with `et18_2026_kinematics.json`.
  - Updated startup profile selection in `main_gui_classic.py` and `main_gui_pro.py` to automatically load and activate `et18_2026_kinematics.json` upon initialization, populating inputs, kinematic plots, and status banners.
  - Configured `PhysicsThread14DOF` and `Interactive14DOFDrivingWindow` in `interactive_driver_14dof.py` to default to `VehicleParams.nominal_fsae_params()`.
- **Dynamic Tire Switching & Multi-Tire Native C ODE Signature Caching**:
  - Implemented parameter-signature caching (`ode_14dof_{sig}.dll` and `ode_14dof_{sig}.sig`) so that switching tires (e.g. Hoosier R20 vs Continental vs 215/60 R17) dynamically compiles and loads the dedicated native C ODE without stale binary clashes.
  - Resolved dynamic vertical wheel load transfer in the simulator under acceleration, braking, and steering maneuvers.
- **Standalone Build & Verification**:
  - Protected `default_vehicle.json` from accidental overwriting during PyInstaller specfile packaging.
  - Rebuilt and verified `dist\MDO_Target_Suite\MDO_Target_Suite.exe` with offscreen process health check.
  - Ran full test suite: 156/156 tests passing.

## [4.5.2] — 2026-09-13

### Added & Optimized (Native C 14-DOF ODE Compilation & Compiler Discovery)
- **Native C 14-DOF Vehicle Dynamics ODE Engine**:
  - Activated and finalized CasADi C-code compilation pipeline in `vehicle_14dof.py`.
  - Added multi-tier compiler detection (`_find_c_compiler`): PATH (`cl`/`gcc`/`clang`), local portable MinGW toolchain (`tools/mingw/bin/gcc.exe`), common MSYS2/UCRT paths (`C:\msys64\ucrt64\bin\gcc.exe`), and Visual Studio MSVC via `vswhere.exe`.
  - Integrated portable MinGW-w64 toolchain in `tools/mingw/` with `.gitignore` exclusion.
  - Implemented Windows 8.3 short-path resolution (`GetShortPathNameW`) in `load_compiled_ode` to eliminate dynamic loading failures (`WIN32 Error 126`) caused by non-ASCII path characters (e.g. `Università`).
  - Added `-shared -fPIC -O3 -ffast-math` compilation flags for GCC, reducing single ODE evaluation time to **~19.3 µs** and full 14-DOF RK4 step to **0.117 ms** (>8500 Hz theoretical calculation rate).
  - Added benchmark and verification suite `test_c_ode_speed.py` validating identical trajectory integration ($8.11\times 10^{-14}$ divergence) against CasADi symbolic `MXFunction`.
  - Updated `MDO_Target_Suite.spec` to automatically bundle `ode_14dof_gen.dll` into PyInstaller binary distributions.

## [4.5.1] — 2026-09-13

### Fixed & Engineered (Studio 3D In-Process, Pitch Damper Damping, Standstill Elastic Clamp, ARB Diamond Ideal/Real & TV Signs)
- **3D Preview Studio In-Process Launching**:
  - Replaced `subprocess.Popen([sys.executable, script_path, ...])` in `main_gui_classic.py` and `interactive_driver_14dof.py` with direct in-process instantiation of `CarPreviewStudioWindow`.
  - Added CLI flag `--preview-3d` in `main_gui.py` to allow direct standalone launch without spawning a duplicate `main_gui_pro`.
  - Added `scripts` directory to `datas` and `scripts.preview_car_3d` to `hiddenimports` in `MDO_Target_Suite.spec`.
- **Öhlins TTX25 Pitch Damping Formulation & Sign Inversion**:
  - Corrected pitch damping kinematics in `vehicle_14dof.py`: aligned wheel vertical velocities $v_{w,\text{pitch}} = a \dot{\theta}$, $v_{w,\text{pitch}} = b \dot{\theta}$ and formulated dissipative pitch moment $M_{\text{pitch,damp}} = 2a F_{w,f} + 2b F_{w,r}$.
  - Completely eliminated self-excited explosive pitch oscillations, matching linear damping behavior $-C_\theta \dot{\theta}$ with guaranteed dissipation.
- **Standstill Ghost Acceleration Elimination**:
  - Fixed relaxation length contact patch elastic recoil in `interactive_driver_14dof.py` at zero throttle (`cmd['throttle'] < 0.01`).
  - Implemented strict standstill clamp at $v < 0.025\text{ m/s}$ (or brake $> 5\%$) clearing $u_x, u_y$ stored strain energy and wheel angular speeds.
  - Applied numerical dissipation ($0.80\times$) below $0.08\text{ m/s}$ to eliminate residual creep.
- **Anti-Roll Bar (ARB) Setup UI & Digitized Curve Diamonds (Ideal 05 & Real 06)**:
  - Added dedicated **"Barre Antirollio (ARB) & Bilanciamento Meccanico"** section in Tab "Assetto & Aerodinamica" of `interactive_driver_14dof.py`.
  - Integrated selectable calculation domains:
    1. *Ideale (05: Telaio Rigido)* from `05_arb_blade_setups__kphi_vs_ms.png`: Soft-Soft ($1000\text{ Nm/deg}, 49\%$), Stiff Ant. ($1370\text{ Nm/deg}, 59.5\%$), Stiff Post. ($1420\text{ Nm/deg}, 38\%$), Stiff-Stiff ($1650\text{ Nm/deg}, 50\%$), ET18 Baseline ($1200\text{ Nm/deg}, 50\%$).
    2. *Reale (06: Torsione Telaio Inclusa)* from `06_arb_blade_setups_kphi_vs_ms.png`: Soft-Soft ($1000\text{ Nm/deg}, 48.4\%$), Stiff Ant. ($1300\text{ Nm/deg}, 49.8\%$), Stiff Post. ($1350\text{ Nm/deg}, 46.7\%$), Stiff-Stiff ($1600\text{ Nm/deg}, 48.2\%$), ET18 Baseline Reale ($1180\text{ Nm/deg}, 48.3\%$).
  - Continuous tuning with sliders for $k_\phi$ and $MS_t$, real-time badge calculating $K_{\phi,f}$, $K_{\phi,r}$, $\text{LLTD}$, and estimated Roll Gradient [$^\circ/\text{g}$], with live synchronization to physics thread parameters `p_vec[1..3]`.
- **Active Torque Vectoring Sign Alignment & Telemetry**:
  - Aligned $r_{\text{target}}$ calculation with chassis frame coordinate conventions using $\delta_{\text{physics}}$ ($\delta < 0 \implies r_{\text{target}} < 0$ in right turn).
  - Ensured correct differential torque split sending more drive torque to the outer wheel ($T_{RL} > T_{RR}$ in right turn) to counteract understeer.
  - Telemetry plot 24 now overlays $r_{\text{target}}$ and $r_{\text{eff}}$ on matching signs.
- **Automated Verification**:
  - Added unit tests for TTX25 pitch damping dissipation, right-turn TV signs, and ARB live tuning in `tests/test_powertrain_tv_and_dampers.py`.
  - Passed all 156 unit tests in suite and full 4-stage automated CI pipeline (`scripts/ci_pipeline.py`).

---

## [4.5.0] — 2026-09-13

### Added & Engineered (Powertrain Differentials & TV, TTX25 4-Way Controls, 3D Ghost Car & OpenGL Display Lists)
- **Powertrain RWD Dual-Motor Differentials & Active Torque Vectoring (TV)**:
  - Added 4 selectable differential modes in `params_vehicle.py`, `vehicle_14dof.py`, and `interactive_driver_14dof.py`:
    1. `DUAL_MOTOR_OPEN`: 50/50 open differential with individual wheel slip limiters.
    2. `VIRTUAL_SPOOL`: 100% locked spool differential applying locking torque $T_{\text{lock}} = K_{\text{spool}} \Delta\omega$ to equalize rear wheel speeds.
    3. `VIRTUAL_SALISBURY_LSD`: Multi-plate clutch pack LSD with static preload ($T_{\text{preload}} = 35\text{ Nm}$) and power/coast ramp ratios ($\mu_{\text{power}} = 0.40$, $\mu_{\text{coast}} = 0.20$) using smooth $\tanh(3.0 \Delta\omega)$ friction saturation.
    4. `ACTIVE_TORQUE_VECTORING`: Closed-loop PI yaw controller tracking linear bicycle model target yaw rate $r_{\text{target}} = \frac{v_x}{L + K_{us} v_x^2} \delta$ saturated to adhesion limit $\frac{\mu g}{v_x}$, with anti-windup clamping to $\pm M_{z,\max} = \pm 500\text{ Nm}$, physical wheel torque clipping $[0, T_{\max}]$, and smooth deactivation below $2\text{ m/s}$.
  - Added UI selector and real-time badge in Tab "Assetto & Aerodinamica", live tuning via `update_live_tuning`, and dedicated telemetry signal preset (`24. Torque Vectoring (Delta Mz, r_target)`) in `FullTelemetryWindow`.
- **Öhlins TTX25 MkII 4-Way GUI Controls & Live Tuning**:
  - Integrated 8 independent 4-way damper registers in GUI: Low-Speed Compression (LSC 0–25 clicks), High-Speed Compression (HSC 0–4.0 turns), Low-Speed Rebound (LSR 0–25 clicks), High-Speed Rebound (HSR 0–4.0 turns) for front and rear axles.
  - Added mode selector (`Lineare`, `Öhlins TTX25 Non-Lineare`, `Custom Dyno Curve`) with thread-safe live tuning and instantaneous fast ODE recompilation.
  - Added live modal damping ratio ($\zeta_f, \zeta_r$) and equivalent wheel damping badge in the sidebar.
- **Continuous Spatial Delta-Lap $\Delta t(s)$ & 3D Ghost Car**:
  - Recorded high-resolution spatial trajectory buffer $(s, x, y, z, \psi, t_{\text{lap}})$ during timed laps.
  - Automatically archived personal best lap upon valid Start/Finish gate crossing.
  - Computed continuous spatial delta-lap $\Delta t(s) = t_{\text{current}} - t_{\text{best}}(s)$ rendered in real-time in the HUD lap card (green `-` if ahead, red `+` if behind).
  - Rendered translucent 3D satin ghost car ($\alpha = 0.35$) smoothly interpolated from the best lap at instantaneous lap time.
- **OpenGL Display Lists Optimization**:
  - Precompiled static circuit geometry (asphalt ribbon, kerbs, centerline, photocell gates) into `glGenLists(1)`, eliminating thousands of Python-to-C OpenGL calls per frame.
  - Precompiled vehicle monocoque and aerodynamic bodies in `vehicle_3d_models.py` (`compile_body_display_list`, `invalidate_display_list`), leaving only wheels and driver for dynamic per-frame transformation.
- **Automated Verification**:
  - Created test suite `tests/test_powertrain_tv_and_dampers.py` validating differential modes, TTX25 non-linear curves, spatial delta sign conventions, and display list attributes.
  - Integrated into CI pipeline with 100% pass rate across all 31 test suites and 153 tests.

---

## [4.4.2] — 2026-09-12

### Added & Improved (Clean 3D Models, UI Telemetry Loader, 4-Wheel Force Analysis & SUV Calibrations)
- **Simplified 3D Geometry Models (`vehicle_3d_models.py`)**:
  - *FSAE Single-Seater*: Restored the classic, streamlined Formula Student aesthetic (clean composite monocoque, sidepods, nosecone, faceted driver helmet, front and rear wings, sleek suspension arms, 5-spoke wheels) guaranteeing high FPS and zero rendering clutter.
  - *B-Segment SUV*: Engineered a clean, solid boxy architectural design ("a rettangoloni") with lower chassis volume + dark sill cladding, greenhouse cabin with tinted glass, crisp front grille with LED headlights, rear LED bar, and 215/60 R17 road wheels.
- **Direct UI Telemetry Import (`FullTelemetryWindow` & Simulator UI)**:
  - Added "Carica Telemetria..." button to the simulator action bar and inside `FullTelemetryWindow` toolbar.
  - Full support for importing `.csv` and `.json` telemetry sessions, automatically rebuilding lap partitions, best lap detection, distance/time horizontal scales, and signal overlays.
- **4-Wheel Component Force Telemetry (Preset 2)**:
  - Added dedicated Preset `"2. Forze Ruote 4 Angoli (Fy, Fx, Fz, Coppie)"` plotting individual force components for all 4 corners overlaid on single plots ($F_y$ for FL, FR, RL, RR; $F_x$ for FL, FR, RL, RR; $F_z$ vertical loads; wheel drive/braking torques). Standardized corner color coding across all force plots (Cyan = FL, Blue = FR, Orange = RL, Red = RR).
- **SUV Braking Torque & Thermal Mass Calibration**:
  - Scaled maximum vehicle braking torque from $1400\text{ Nm}$ to $4800\text{ Nm}$ in `params_vehicle.py` (`segmento_b_suv()`) and `setups/segmento_b_suv.json`, delivering $> 1.15\text{ g}$ threshold braking deceleration on the 1250 kg SUV platform.
  - Scaled road tire thermal mass to $C_{\text{th}} = 16000\text{ J/K}$ (~8x Hoosier racing slick) and convective cooling coefficient to $h_{\text{conv}} = 32.0\text{ W/K}$ with reduced hysteretic heat generation (`heat_gen_efficiency = 0.22`), preventing premature thermal runaway and stabilizing running temperatures within $45^\circ\text{C} - 65^\circ\text{C}$.

---

## [4.4.1] — 2026-09-12

### Added & Calibrated (Dedicated Road Tire Model)
- **Dedicated Pacejka MF6.1 Road Touring Tire (`Road_215_60R17_Touring.tir`)**: Full FITTYP=61 model tailored for road SUV vehicles ($F_{z0} = 3500\text{ N}$, $R_w = 0.315\text{ m}$, $R_0 = 0.3449\text{ m}$, $P_{\text{nom}} = 2.3\text{ bar}$). Replaces Hoosier slick scaling by fixing cornering stiffness ($K_{y\alpha} \approx 830\text{ N/deg}$), relaxation lengths ($\sigma_x = 0.28\text{ m}$, $\sigma_y = 0.35\text{ m}$), aligning torque peak ($52.2\text{ Nm}$), and road peak friction ($\mu_y \approx 0.98\text{ g}$, $\mu_x \approx 1.05\text{ g}$).
- **Automatic Tire Setup Binding**: Added `tire_file` field to `VehicleParams` dataclass. Selecting `segmento_b_suv.json` or switching setups automatically binds and loads `Road_215_60R17_Touring.tir` across both MDO workspaces and the real-time 14-DOF simulator.
- **Unit & Invariant Tests**: Added `test_06_road_tire_file_properties` in `tests/test_b_segment_suv.py` verifying nominal load, dimensional geometry, and MF6.1 non-linear force generation.

---

## [4.4.0] — 2026-09-12

### Added & Architecture (B-Segment FWD SUV, Open Differential & 3D Models)
- **B-Segment Compact SUV Setup & Vehicle Profile**: Added `VehicleParams.segmento_b_suv()` and `setups/segmento_b_suv.json` with realistic parameters ($1250\text{ kg}$, $60/40$ weight distribution, $h_{cg} = 0.56\text{ m}$, soft road suspension $K_{z} \sim 28\text{ kN/m}$, $\pm 80\text{ mm}$ travel, $15.5:1$ steering ratio).
- **FWD Drivetrain & Mechanical Open Differential**: Implemented front-wheel drive torque routing and inner-wheel spin limiter in 14-DOF physics ODEs (`vehicle_14dof.py` and `interactive_driver_14dof.py`).
- **Modular 3D Car Rendering (`vehicle_3d_models.py`)**: Structured vehicle renderers into `FSAECar3DModel` and `SUVCar3DModel` (solid SUV body, skid plates, grille, roof rails, LED lights, 17" alloy wheels).
- **Interactive 3D Car Preview Studio (`scripts/preview_car_3d.py`)**: Standalone orbital 3D studio with live suspension/steering controls, grip utilization HUD, and vehicle selector. Integrated into GUI tools menu and simulator setup panel.
- **Wheel Force & Drive/Brake Torque Telemetry**: Added 4-wheel telemetry panels for $F_x, F_y, F_z$ wheel forces and wheel-by-wheel drive/braking torques.

---

## [4.1.0] — 2026-09-11

### Added & Improved (Simulator, Steering Kinematics & Tire Wear)
- **High-Clarity Circuit HUD Minimap**: Integrated real-time track minimap card in `interactive_driver_14dof.py` HUD (`OpenGLViewport3D` & `_LegacyViewport3D`). Renders scaled track ribbon, S/F line marker, progress percentage/distance, and dynamic directional vehicle arrow oriented with instantaneous car yaw angle $\psi$. Toggable via hotkey `M` and HUD options.
- **Adams Car Wheel Steer Kinematics (`wheel_steer_f_coeffs`)**: Replaced the singular, asymptotic Ackermann % formula with direct polynomial mapping derived from multibody Adams Car simulations (`wheel_steer_vs_pinion_steer`). Handled bilateral mirror symmetry $\delta_{FL}(\delta_{sw}) = f(\delta_{sw})$, $\delta_{FR}(\delta_{sw}) = -f(-\delta_{sw})$. Guaranteed that the inside wheel turns sharper than the outside wheel ($\Delta\delta > +2.8^\circ$ at $100^\circ$ lock) both mathematically in CasADi ODEs and visually in 3D/2D viewports, resolving the visual "anti-Ackermann" appearance.
- **Tire Wear Degradation Recalibration**: Corrected the dimensionless wear coefficient in `tire_degradation.py` from $2.5 \times 10^{-10}$ to calibrated physical FSAE Hoosier R20 value $k_{\text{wear}} = 2.0 \times 10^{-3}$, providing realistic $\sim 3\% - 8\%$ wear after 5 km and $\sim 20\% - 35\%$ over full 22 km endurance. Added dynamic acceleration selector `combo_wear_scale` (`1x Reale Endurance`, `2x`, `5x Test Rapido`, `10x Sprint Usura`) in simulator UI and `SimulationOptions`. Set degradation mode to "Leggero (1-Zona Bulk)" by default for immediate feedback.
- **New Unit Tests**: Added `tests/test_ackermann_wear.py` validating inside-greater-than-outside steering, bilateral mirror symmetry, 5 km wear accumulation, and smoke rendering of the minimap.

---

## [4.0.1] — 2026-09-11

### Fixed & Improved (Interactive 14-DOF Simulator)
- **Direct 1:1 Steering Authority**: Removed artificial speed-based steering angle attenuation (`speed_factor = 1.0` by default), eliminating the issue where accelerating stripped away steering angle and caused severe understeer. Increased steering rate limiter to $720^\circ/\text{s}$ for crisp countersteering response. Added an optional gentle assist checkbox (`Assistenza Attenuazione Sterzo con Velocità`) in settings.
- **Resizable Sidebar & UI Layout**: Replaced fixed `QHBoxLayout` with an interactive `QSplitter(Qt.Horizontal)`, removing `right_panel.setMaximumWidth(460)`. Users can now freely resize the sidebar panel to accommodate all customization controls, sliders, and telemetry graphs.
- **SpinBox Keyboard Focus & Hotkey Isolation**: Fixed bug in `keyPressEvent` and `keyReleaseEvent` where typing into spinboxes (`sb_ffb_gain`, etc.) focused the inner `QLineEdit` and triggered camera switches or resets. The focus check now properly recognizes `QLineEdit`, `QAbstractSpinBox`, and parent spinboxes.
- **Clean Label Formatting**: Eliminated Qt mnemonic accelerator underscores caused by single `&` characters in Tab 2 (`"Calibrazione Controller e Fisica"`) and GroupBox titles.
- **Force Feedback & Trigger Haptics Fix**: Resolved the critical bug where enabling trigger haptics shut down all force feedback. The code previously attempted to call undocumented XInput ordinal 100 assuming it was `XInputSetStateEx` (in reality `XInputGetStateEx`), bypassing standard `XInputSetState`. Restored direct `XInputSetState` and engineered responsive rumble blending: wheel lockup / ABS shudder (brake LT) now directly rumbles the left grip motor, while traction loss / wheelspin buzz (throttle RT) directly vibrates the right grip motor, guaranteeing distinct tactile pedal feedback on all controllers.
- **Cockpit POV Camera**: Elevated driver perspective ($x = +0.24\text{ m}, z = +0.44\text{ m}$) above the cockpit rim looking cleanly down the nosecone, preventing internal chassis mesh clipping during acceleration squat.

---

## [4.0.0] — 2026-09-10

### Added
- **GitHub Actions CI Workflow** (`.github/workflows/ci.yml`): Automated multi-step CI pipeline running physics tests, GUI lifecycle tests, and standalone PyInstaller builds on Windows.
- **Physical Invariant Tests**:
  - `test_14dof_openmdao_equilibrium_and_roll_gradient`: Tight regression test bounded by analytical formula $\phi/a_y = m \cdot h_{\text{arm}} / K_{\phi,\text{tot}} \approx 2.44\text{ deg/g}$ ($\pm 30\%$).
  - `test_14dof_understeer_gradient_weight_distribution_sensitivity`: Verified monotonic response of $K_{us}$ to weight distribution ($wd_f$).
  - `test_14dof_yaw_rate_monotonic_buildup`: Verifies positive damping and monotonic yaw rate ramp-up during initial step-steer excitation.
  - `test_14dof_drag_glide_energy_conservation`: Verifies monotonic speed decay driven purely by aerodynamic drag in straight glide.
  - `test_inflation_pressure_sensitivity` & `test_zero_vertical_load_gives_zero_forces` in `tests/test_tire_mf61.py`.
- **Benchmark Performance Tests** (`tests/test_benchmark_performance.py`):
  - 14-DOF Trim Residual Evaluation ($< 5.0\text{ ms}$).
  - Vehicle DNA Maps 2D computation ($< 250.0\text{ ms}$).
  - 14-DOF Step-Steer 2.0s ODE integration ($< 2000.0\text{ ms}$).
- **Pytest Configuration** (`pytest.ini`): Clean test discovery with registered `benchmark` and `gui` markers.
- **Modular GUI Architecture**:
  - `gui_canvas.py`: Dedicated Matplotlib `FigureCanvasQTAgg` wrapper (`MplCanvas`).
  - `gui_dialogs.py`: `ChartGuideDialog`, `PopoutPlotDialog`, and `translate_figure_language` engine with 140+ SAE/academic terms.
  - `gui_workers.py`: `SensitivityWorker`, `ARBDiamondWorker`, `CollocationWorker`, and multiprocessing speed-sweep routines.
- **Open-Source License**: Standard MIT License (`LICENSE`).

### Changed
- **Vehicle Parameterization & Base Setup** (`params_vehicle.py`, `setups/default_vehicle.json`):
  - Calibrated default vehicle roll stiffness to exact FSAE targets: $K_{\phi,f} = 470\text{ Nm/deg}$ ($26929\text{ Nm/rad}$) and $K_{\phi,r} = 530\text{ Nm/deg}$ ($30367\text{ Nm/rad}$), totaling $1000\text{ Nm/deg}$ ($57296\text{ Nm/rad}$, LLTD 47.0%).
  - Enhanced `suspension_sizing.py` to calculate explicit axle roll stiffness components: springs ($K_{\phi,\text{spring}}$), anti-roll bars ($K_{\phi,\text{arb}}$), tire vertical radial compliance ($K_{\phi,\text{tire}} = \frac{1}{2}K_t t^2$), and effective series stiffness ($K_{\phi,\text{series}} = \frac{K_{\text{susp}} K_{\text{tire}}}{K_{\text{susp}} + K_{\text{tire}}}$).
  - Declared `Ix: float = 35.0` and `Iy: float = 110.0` as first-class dataclass fields.
  - Declared `spindle_offset_f: float = 0.12` and `spindle_offset_r: float = 0.10`.
  - Replaced hard-coded constants in `vehicle_14dof.py` with `vp.Ix` and `vp.Iy`.
  - Documented suspension dynamic relaxation parameter $\tau_z = 0.02\text{ s}$ (~50 Hz).
  - Consolidated 7 duplicate local declarations of $g = 9.81$ into `vp.g`.
- **Packaging & Dependencies** (`requirements.txt`):
  - Pinned exact compatibility ranges (`~=` and `>=, <next_major`) for `casadi`, `openmdao`, `numpy`, `scipy`, `matplotlib`, `PyQt5`, `pygame`, `pandas`, and `pytest`.
- **PyInstaller Specification** (`MDO_Target_Suite.spec`):
  - Added `gui_canvas`, `gui_dialogs`, and `gui_workers` to `hiddenimports`.
- **Developer Documentation** (`docs/DEVELOPER_GUIDE.md`):
  - Added dedicated sections on Test Architecture, Physical Invariants, and CI/CD.

### Removed
- Removed obsolete throwaway patch script `modify.py`.
- Removed deprecated 33-line toy model `tire_model_planar.py`.
- Removed 20 stale OpenMDAO output directories (`__main__*_out/`, `problem_out/`).
- Removed runtime crash dumps, temporary test plots, and raw notebook text dumps from `scratch/`.

### Relocated
- Moved `check_env.py` $\to$ `scripts/check_env.py`.
- Moved `check_partials_n2.py` $\to$ `scripts/check_partials_n2.py`.

---

## [3.9.5] — 2026-09-08
- Roll-Angle Dependent Roll Centers ($h_{rc}(\phi)$ and $y_{rc}(\phi)$).
- Active Vehicle Parameters Inspector Dialog (`vehicle_parameters_inspector_dialog.py`).
- 15 Adams Car kinematic digitized curves (ET-17 2025 and ET-18 2026 presets).

---

## [3.0.0] — 2026-08-15
- Second generation professional UI: `main_gui_pro.py` with 5 engineering workspaces.
- Full 14-DOF Real-Time Gamepad Driving Simulator (`interactive_driver_14dof.py`).
- Vehicle DNA Maps & Standard Understeer Gradient formulation.