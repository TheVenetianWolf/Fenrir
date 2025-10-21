# 🔐 Security Policy

Thanks for helping keep FENRIR safe for educators and researchers.

FENRIR is an **educational simulation** of pursuit/guidance.  
It is **not** an operational system and does not control real hardware.  
Still, we take security, privacy, and supply-chain hygiene seriously.

---

## 📣 Reporting a Vulnerability

- Use **GitHub Private Vulnerability Reporting** (preferred):  
  Go to the repo → **Security** → **Report a vulnerability**.
- Or email the maintainer: lupus.wildcard210@aleeas.com.

Please include:
- A clear description and steps to reproduce
- A minimal proof of concept (if applicable)
- Affected version/commit hash and your environment

We’ll acknowledge within **72 hours** and aim to provide a fix or mitigation timeline within **7 days** for high-impact issues.

Safe harbor: **Good-faith research is welcome**. We will not pursue legal action for security research that respects privacy, keeps data confidential, and avoids service disruption.

---

## 🔭 Scope

In scope:
- Vulnerabilities in this repository’s code (simulation engine, UI, docs site if any)
- Supply-chain issues in our dependencies (e.g., malicious packages)
- Accidental secrets or tokens committed to the repo (please report immediately)

Out of scope:
- Attacks against third-party services (e.g., Streamlit Cloud, GitHub)
- Real-world weapons control systems (FENRIR is **simulation only**)
- Feature requests or physics inaccuracies (use Issues/Discussions)

---

## 🔄 Supported Versions

FENRIR is evolving quickly. The **latest `main`** and the **most recent tagged release** are supported for security fixes.

---

## 🔑 Secrets & Keys

This project does **not** require API keys or credentials.  
If you ever find a secret (token, password) in history or issues, **report it privately** so we can rotate and purge.

---

## 🔧 Dependencies

We use pinned dependencies where practical and monitor CVEs via **Dependabot**.  
If you spot a vulnerable transitive dependency, please open a report with the package name and version.

---

## ⚠️ Responsible Use

This project is intended for **education and research**.  
Do not use FENRIR to harm people or property. Follow applicable laws and export controls.
