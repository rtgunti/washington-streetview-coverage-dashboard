# Washington imagery coverage dashboard

This is a self-contained Streamlit Community Cloud repository for comparing
Mapillary and KartaView road coverage across Washington State from 2020–2026.

## Local verification

From this directory:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Create a new app using this repository's `main` branch and the following
entrypoint:

```text
app.py
```

The app does not require Mapillary or KartaView API credentials. All runtime data
is included under `assets/`. The largest individual file is below GitHub's 100 MiB
normal-file limit.

The compact FlatGeobuf files contain deployment-ready Overture road coverage.
The GeoPackage contains provider sequence tracks used only for coordinate-level
inspection. Raw API caches, canonical road-segment Parquet, and the full Overture
road warehouse are intentionally excluded.
