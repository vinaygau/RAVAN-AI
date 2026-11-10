# Ravan AI Trajectory Review Dashboard

The dashboard reviews the Phase 6 held-out test episodes and connects each time sample to the Phase 6 score, Phase 7 recommendation, and Phase 8 model explanation.

From the project root, rebuild the joined data bundle with:

```powershell
python scripts/build_phase9_dashboard.py
```

Then serve the project files locally:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/dashboard/`. The page and its data bundle are local; the dashboard makes no network calls. Labels are shown for research review and are not live predictions or safety guarantees.
