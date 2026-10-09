import os
import sys
import time
import ctypes
from ctypes import wintypes
import subprocess

class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [
        ('wLeftMotorSpeed', wintypes.WORD),
        ('wRightMotorSpeed', wintypes.WORD)
    ]

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', wintypes.WORD),
        ('bLeftTrigger', wintypes.BYTE),
        ('bRightTrigger', wintypes.BYTE),
        ('sThumbLX', wintypes.SHORT),
        ('sThumbLY', wintypes.SHORT),
        ('sThumbRX', wintypes.SHORT),
        ('sThumbRY', wintypes.SHORT),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [('dwPacketNumber', wintypes.DWORD), ('Gamepad', XINPUT_GAMEPAD)]

def run_diagnostics():
    print("=" * 65)
    print("   DIAGNOSTICA AVANZATA VIBRAZIONE E TRIGGER HAPTICS (USB)")
    print("=" * 65)

    xinput = None
    for dll in ['xinput1_4.dll', 'xinput1_3.dll', 'xinput9_1_0.dll']:
        try:
            xinput = ctypes.windll.LoadLibrary(dll)
            print(f"[OK] Libreria XInput caricata: {dll}")
            break
        except Exception:
            continue

    if not xinput:
        print("[ERR] Impossibile caricare XInput su questo sistema.")
        return 1

    active_ports = []
    state = XINPUT_STATE()

    print("\n--> Scansione porte XInput...")
    for p in range(4):
        res = xinput.XInputGetState(p, ctypes.byref(state))
        if res == 0:
            print(f"    [PORTA {p}] Controller CONNESSO e ATTIVO")
            active_ports.append(p)
        else:
            print(f"    [PORTA {p}] Non connesso (Codice errore: {res})")

    if not active_ports:
        print("\n[ERR] Nessuna porta XInput risponde. Controlla che il controller sia acceso e collegato.")
        return 1

    target_port = active_ports[0]
    print(f"\n[TARGET] Utilizzo Porta {target_port} per i test.")

    print("\n" + "-" * 50)
    print("TEST 1: MOTORI PALMARI VIA XINPUT")
    print("-" * 50)

    def set_xinput_vib(port, left, right):
        v = XINPUT_VIBRATION(int(left), int(right))
        return xinput.XInputSetState(port, ctypes.byref(v))

    print("--> 1.1 Motore Palmare SINISTRO (Bassa frequenza / Heavy) per 1 sec...")
    ret = set_xinput_vib(target_port, 65535, 0)
    print(f"    Chiamata Win32: codice {ret} (0 = successo)")
    time.sleep(1.0)
    set_xinput_vib(target_port, 0, 0)
    time.sleep(0.3)

    print("--> 1.2 Motore Palmare DESTRO (Alta frequenza / Light) per 1 sec...")
    ret = set_xinput_vib(target_port, 0, 65535)
    print(f"    Chiamata Win32: codice {ret} (0 = successo)")
    time.sleep(1.0)
    set_xinput_vib(target_port, 0, 0)
    time.sleep(0.3)

    print("--> 1.3 ENTRAMBI i motori palmari al 100% per 1 sec...")
    ret = set_xinput_vib(target_port, 65535, 65535)
    print(f"    Chiamata Win32: codice {ret} (0 = successo)")
    time.sleep(1.0)
    set_xinput_vib(target_port, 0, 0)
    time.sleep(0.5)

    print("\n" + "-" * 50)
    print("TEST 2: VIBRAZIONE NATIVA PYGAME (SDL RUMBLE)")
    print("-" * 50)
    try:
        import pygame
        pygame.init()
        pygame.joystick.init()
        n_joy = pygame.joystick.get_count()
        print(f"[OK] Joystick rilevati da Pygame: {n_joy}")
        if n_joy > 0:
            js = pygame.joystick.Joystick(0)
            js.init()
            print(f"    Nome: {js.get_name()}")
            print("--> Invio impulso Pygame rumble (100% per 800ms)...")
            if hasattr(js, 'rumble'):
                ok = js.rumble(1.0, 1.0, 800)
                print(f"    Esito js.rumble: {ok}")
                time.sleep(1.0)
            else:
                print("    [!] La versione Pygame in uso non supporta js.rumble()")
    except Exception as e:
        print(f"    [ERR] Errore test Pygame: {e}")

    print("\n" + "-" * 50)
    print("TEST 3: GRILLETTI FISICI (IMPULSE TRIGGERS) VIA WINDOWS RUNTIME")
    print("-" * 50)
    print("Questo test pilota direttamente l'API di sistema Windows.Gaming.Input.")

    ps_script = """
[Windows.Gaming.Input.Gamepad, Windows.Gaming.Input, ContentType=WindowsRuntime] | Out-Null
$pads = [Windows.Gaming.Input.Gamepad]::Gamepads
if ($pads.Count -eq 0) {
    Write-Output "NO_PAD_WGI"
    exit
}
$pad = $pads[0]
$v = New-Object Windows.Gaming.Input.GamepadVibration

# Test Trigger Sinistro (Freno)
Write-Output "VIB_LEFT_TRIGGER"
$v.LeftTrigger = 1.0; $v.RightTrigger = 0.0; $v.LeftMotor = 0.0; $v.RightMotor = 0.0
$pad.Vibration = $v
Start-Sleep -Milliseconds 1000

# Test Trigger Destro (Gas)
Write-Output "VIB_RIGHT_TRIGGER"
$v.LeftTrigger = 0.0; $v.RightTrigger = 1.0; $v.LeftMotor = 0.0; $v.RightMotor = 0.0
$pad.Vibration = $v
Start-Sleep -Milliseconds 1000

# Stop
$v.LeftTrigger = 0.0; $v.RightTrigger = 0.0; $v.LeftMotor = 0.0; $v.RightMotor = 0.0
$pad.Vibration = $v
Write-Output "DONE_WGI"
"""

    try:
        print("--> Esecuzione comando PowerShell WGI...")
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        out = proc.stdout.strip()
        print("    Output WGI:")
        for line in out.splitlines():
            print(f"      {line}")
        if "NO_PAD_WGI" in out:
            print("    [!] Windows.Gaming.Input non ha trovato controller registrati.")
        elif "DONE_WGI" in out:
            print("    [OK] Sequenza Impulse Triggers inviata con successo all'hardware.")
    except Exception as e:
        print(f"    [ERR] Errore esecuzione PowerShell WGI: {e}")

    print("\n" + "=" * 65)
    print("   DIAGNOSTICA COMPLETATA")
    print("=" * 65)
    return 0

if __name__ == '__main__':
    sys.exit(run_diagnostics())