# 🐺 Contributing to FENRIR

First off, thank you for your interest in contributing!  
FENRIR began as a small curiosity project over a caffeine-fueled weekend and grew into a full open-source simulation tool for exploring pursuit and guidance physics.  
Contributions, ideas, and improvements are always welcome.

---

## 🧭 Philosophy

FENRIR’s purpose is to make **complex motion dynamics** understandable and **interactive**.  
Whether your focus is on maths, physics, UI design, or storytelling through code — your perspective adds value.

The tone of this project is **collaborative and educational**.  
Good science, clear code, and curiosity are the guiding principles.

---

## 🧰 Getting Started

1. Fork the repository
   ```bash
   git clone https://github.com/<your-username>/Fenrir.git
   cd Fenrir
   ```
2. Set up your environment

  FENRIR uses Poetry for dependency management.
  ```bash
  poetry install
  poetry shell
  ```
  If you prefer pip:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # macOS/Linux
  pip install -r requirements.txt
  ```

3. Run the dashboard
  ```bash
  streamlit run ui/streamlit_app.py
  ```

## 🧠 How to Contribute

You can help by:

- Improving documentation or adding physics explanations
- Optimising simulation code or adding new dynamics models
- Enhancing UI / UX of the Streamlit dashboard
- Adding test cases (pytest is recommended)
- Translating or simplifying the physics for education
- Reporting bugs or requesting new features under Issues

  When you make changes, please:
  1. Create a branch (feature/radar-upgrade, fix/physics-doc, etc.)
  2. Commit clearly:

  ```bash
  git commit -m "Add radar visual sweep effect"  # concise, imperative mood
  ```

  3. Push and open a Pull Request.

 ## 🧪 Code Style & Testing
 - Follow PEP8 standards.
- Use docstrings (triple quotes) for functions and classes.
- Keep code readable and well-commented, FENRIR is also a learning tool.
- Before opening a PR:

  ```bash
  poetry run pytest
  ```
 - Include short explanations for complex maths.

## 🧩 File Overview
| Folder      | Purpose                                     |
| ----------- | ------------------------------------------- |
| `core/`     | Physics engine, dynamics, entities          |
| `services/` | Simulation scenarios & behaviours           |
| `ui/`       | Streamlit interface                         |
| `tests/`    | Unit tests                                  |
| `assets/`   | Logos or visuals                            |
| `docs/`     | Optional technical or physics documentation |

## 🐾 Code of Conduct
All contributors are expected to follow respectful, inclusive behaviour.
No hostility, no ego. We’re here to learn, share, and occasionally curse at floating-point errors together.

## 📬 Questions?
- Open a GitHub issue.
- Or start a Discussion if your idea is exploratory

---
FENRIR is about curiosity meeting clarity.
If you learned something while contributing, that’s already a win.

Thank you for helping this project evolve 🐺
