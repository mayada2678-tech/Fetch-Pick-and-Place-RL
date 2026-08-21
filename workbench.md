%%writefile Mayada/fetch_pick_and_place/workbench.md
# Fetch Pick And Place Workbench

## Überblick

Diese Workbench ist eine eigenständige Stable-Baselines3-Anwendung für eine Fetch Pick And Place Umgebung (z.B. `FetchPickAndPlace-v1`). Sie enthält dieselbe grundlegende Struktur wie die HalfCheetah-V5-Version: GUI, Training, Live-Animation, Reward-Plot, Vergleichsmodus und Modell-Management.

Die Standard-Methoden in dieser Workbench sind `SAC`, `TD3` und `PPO`. Die Umgebung ist ein MuJoCo-Task und benötigt eine valide MuJoCo-Installation mit kompatiblem Interpreter.

---

## Installation

Im Ordner `Mayada/fetch_pick_and_place`:

```powershell
python -m pip install -r requirements.txt
```

Abhängigkeiten:

- `gymnasium[mujoco]>=0.29.1`
- `mujoco>=3.0.0`
- `stable-baselines3>=2.3.0`
- `torch>=2.2.0`
- `matplotlib>=3.7.0`
- `numpy>=1.24.0`
- `gymnasium-robotics`

---

## Start

```powershell
python fetch_pick_and_place_app.py
```

---

## Unterstützte Methoden

- `SAC`
- `TD3`
- `PPO`

Die Auswahl erfolgt im UI über die aktive Methode und im Methodenfenster.

---

## Funktionen

- Training für `FetchPickAndPlace-v1`
- Hyperparameter pro Methode
- Vergleich mehrerer Methoden
- Live-Animation während des Trainings und Vergleichs
- Parameter-Sweep pro Methode mit eigener Auswahl von Lernmethode und Parameter
- Reward-Verlauf und Vollplot
- Pause/Resume und Abbruch
- Modell speichern und laden
- automatische Qualitätsbewertung
- Dark/Light Theme

### Animationsstatus

Die Live-Animation ist für Trainings- und Vergleichsläufe aktiv, sofern die Umgebung einen gültigen Render-Kontext liefert. Die GUI schützt die Render-Callbacks gegen fehlende Initialisierung und hält die Animation auch bei Sweep- und Vergleichsläufen stabil aktiv.

---

## Projektstruktur

```text
fetch_pick_and_place/
├── fetch_pick_and_place_app.py
├── fetch_pick_and_place_gui.py
├── fetch_pick_and_place_logic.py
├── workbench_base.py
├── workbench.md
├── requirements.txt
├── saved_models/
└── test_fetch_pick_and_place_logic.py
```

---

## Wichtige Logik

Die Datei `fetch_pick_and_place_logic.py` enthält:

- `SUPPORTED_METHODS = ("SAC", "TD3", "PPO")`
- `FETCH_PICK_AND_PLACE_ENV_ID = "FetchPickAndPlace-v1"`
- `OnPolicyConfig`
- `get_default_parameters_for_method(...)`
- `SyncVectorEnvAdapter`
- `TrainingCallback`
- `FetchPickAndPlaceTrainer`
- Qualitätsbewertung mit `assess_training_quality(...)`

Dabei gilt als Erfolgsmaßstab ein mittlerer Reward ab einem definierten Solved-Wert. Das genaue Muster entspricht der Hopper-V5-Implementation.

---

## Tests

```powershell
python -m pytest test_fetch_pick_and_place_logic.py -q
```

---

## Hinweis

Die MuJoCo-Umgebung ist anspruchsvoll; die Workbench prüft die Laufzeitumgebung beim Start des Trainings und zeigt einen klaren Fehler an, wenn MuJoCo auf der aktuellen Maschine nicht nutzbar ist.

## Troubleshooting

### Live-Animation funktioniert nicht

- Auf einem Headless-Server oder in einer Remote-Umgebung fehlt oft ein gültiger GUI-/OpenGL-Kontext.
- Verwende die Workbench auf einem lokalen Desktop mit einer echten Fensterumgebung.
- Wenn kein gültiger Render-Kontext verfügbar ist, deaktiviert die App die Animation automatisch und läuft trotzdem weiter.

### MuJoCo-Fehler beim Start

- Stelle sicher, dass der gewählte Python-Interpreter mit `mujoco`, `gymnasium`, `gymnasium-robotics` und `stable-baselines3` kompatibel ist.
- Prüfe, ob die Umgebung korrekt aktiviert ist.
- Wenn erforderlich, nutze die im Projekt hinterlegte `requirements.txt` und installiere sie erneut.

### Render-Kontext in Remote-/VM-Umgebungen

- Wenn der Code in einer VM, WSL-Instanz oder Remote-Desktop-Umgebung läuft, kann der OpenGL-/MuJoCo-Render fehlschlagen.
- In solchen Fällen ist der Trainingslauf ohne Animation oft der robusteste Modus.