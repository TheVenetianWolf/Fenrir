# 🐺 FENRIR: Defense Simulation Dashboard

**FENRIR** is an open-source, real-time missile guidance simulation dashboard built with **Python** and **Streamlit**.  
It visualises pursuit and intercept dynamics in a clean, interactive interface designed for education, research, and experimentation.

---

## 🚀 Features

- **Proportional Navigation (PN) Intercept** simulation with adjustable parameters  
- Optional **Evasive Target** behaviour for testing guidance robustness  
- **Top-Down Trajectory** view (live-updating)  
- **Telemetry charts** for range, LOS rate, and acceleration over time  
- **Data export** to CSV for analysis  
- **Real-time dashboard** built entirely in Python using Streamlit  
- Modular backend for future extensions (e.g., seeker models, autopilot lag, radar view)

---

## 🧠 Concept

FENRIR demonstrates how simple missile guidance laws behave in dynamic pursuit.  
The simulation integrates missile and target kinematics, updating in real time as parameters (navigation constant `N`, max acceleration, time step, etc.) are changed.

The name **FENRIR** is inspired by the mythic wolf from Norse legend, a symbol of relentless pursuit.

---

## 🧩 Project Structure

```bash
Fenrir/
│
├── core/                  # Physics and entity definitions
│   ├── engine.py
│   └── entities.py
│
├── services/              # Scenario definitions and helpers
│   ├── scenarios.py
│   └── ...
│
├── ui/                    # User interface (Streamlit app)
│   └── streamlit_app.py
│
├── run.py                 # Entry point for local testing
├── pyproject.toml         # Poetry project file
├── requirements.txt       # Deployment dependencies
└── README.md
```
---

## 💻 Getting Started

### Prerequisites
- Python ≥ 3.10  
- Git
- Either [Poetry](https://python-poetry.org/) or `pip`  
- Streamlit, Matplotlib, NumPy, Pandas

```text
Windows tip: Use PowerShell as admin for installs.
macOS tip: If using Homebrew Python, prefer python3 and pip3.
```

### Installation

#### Pick a folder you like, then:
```bash
git clone https://github.com/TheVenetianWolf/Fenrir.git
cd Fenrir
```

#### Install dependencies into a virtualenv Poetry manages for you
```bash
poetry install
```

### Run Dashboard
```bash
poetry run streamlit run ui/streamlit_app.py
```

Then Open URL provided (Usually streamlit opens it automatically)

### For pip and venv
```bash
# Create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt  # If present
# If not present, install the core deps:
# pip install streamlit numpy pandas matplotlib scipy
```

### Run the app
```bash
streamlit run ui/streamlit_app.py
```

## Contributing
Pull requests and ideas are welcome!
Please fork the repository and submit a PR with clear commit messages.
Discussions about physics, simulation fidelity, or UX are encouraged.

## License
This project is released under the MIT License (see LICENSE)

## Citation
If you use FENRIR in research or teaching, please credit it as:
```text
FENRIR: Open-Source Defense Simulation Dashboard, 2025
https://github.com/TheVenetianWolf/Fenrir
```

## 🐾 Acknowledgements
Inspired by classical missile guidance theory and the curiosity to visualise physics beautifully.