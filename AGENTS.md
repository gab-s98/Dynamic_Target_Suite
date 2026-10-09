# Dynamic Target Suite — Regole di Progetto & Linee Guida di Automazione (AGENTS.md)

Questo file definisce le regole e le istruzioni di automazione che gli agenti AI e i tool devono seguire all'interno del repository **Dynamic Target Suite**.

---

## 1. Ambiente di Esecuzione (Python Environment)
- Utilizzare **esclusivamente** l'interprete dell'ambiente virtuale locale:  
  `.\venv\Scripts\python.exe`
- Non utilizzare mai l'interprete `python` globale, poiché le dipendenze specialistiche (**CasADi**, **OpenMDAO**, **PyQt5**, **SciPy**, **Pygame**) sono installate all'interno di `.\venv`.

---

## 2. Protocollo di Verifica Automatica (Continuous Integration)
Dopo ogni modifica a moduli fisici o grafici, eseguire la suite di test prima di confermare il completamento:
```bash
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
In alternativa, eseguire lo script unificato di pipeline:
```bash
.\venv\Scripts\python.exe scripts/ci_pipeline.py
```

---

## 3. Gestione Eseguibile Standalone (PyInstaller)
- Mantenere sempre aggiornato [`MDO_Target_Suite.spec`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/MDO_Target_Suite.spec) includendo nei `hiddenimports` i nuovi moduli creati.
- Ricompilare l'eseguibile quando richiesto o in seguito a modifiche strutturali:
  ```bash
  .\venv\Scripts\pyinstaller.exe --noconfirm MDO_Target_Suite.spec
  ```
- Verificare sempre che il binario generato `dist\MDO_Target_Suite\MDO_Target_Suite.exe` si avvii correttamente senza errori di import o DLL mancanti.

---

## 4. Standard Matematici e Documentazione
- **Convenzione Standard Gradiente di Sottosterzo**: La relazione fondamentale per il gradiente delle mappe stazionarie è $\rho_y = -K/L$, da cui discende il gradiente di sottosterzo $K = -L\rho_y$ e la condizione di stabilità $\rho_y < 1/u^2$ (conforme a standard SAE J670e / ISO 4138).
- **Documentazione**: Mantenere allineato il file [`docs/DEVELOPER_GUIDE.md`](file:///c:/Users/light/Documents/Università/E18/Dynamic_Target_Suite/docs/DEVELOPER_GUIDE.md) per ogni nuova funzionalità o modulo aggiunto alla suite.
