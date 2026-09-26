PY ?= .venv/bin/python
PYTHON_BOOTSTRAP ?= $(shell command -v python3.11 || command -v python3.12 || command -v python3)
PORT ?= 8501

.PHONY: setup demo data pull store inventory fit tabpfn test

setup: .venv/.installed

.venv/.installed: requirements.txt
	$(PYTHON_BOOTSTRAP) -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r requirements.txt
	touch $@

# Run the app from committed data/processed/ (no API key, no data download). Open http://localhost:$(PORT)
demo: setup
	$(PY) -m streamlit run app/app.py --server.port $(PORT) --server.address 127.0.0.1

# Pull raw public data (network) and rebuild data/processed/. Not needed to run the demo.
data: setup pull store inventory

pull:
	$(PY) -m src.pull.counties
	$(PY) -m src.pull.hurdat
	$(PY) -m src.pull.wsp
	$(PY) -m src.pull.psurge
	$(PY) -m src.pull.eaglei
	$(PY) -m src.pull.assets
	$(PY) -m src.pull.fema
	$(PY) -m src.pull.hwm

store:
	$(PY) -m src.store.advisory_store
	$(PY) -m src.store.outages
	$(PY) -m src.store.registry
	$(PY) -m src.store.psurge_sites

inventory:
	$(PY) -m src.pull.inventory

# Refit C1 (LightGBM + GLM) on Ian + Idalia and regenerate the backtest report (Milton held out).
fit: setup
	$(PY) -m src.models.c1_fit
	$(PY) -m src.models.backtest

# Optional TabPFN v2 benchmark column (needs torch: requirements-optional.txt). Re-run `make fit` after.
tabpfn: setup
	.venv/bin/pip install -q -r requirements-optional.txt
	$(PY) -m src.models.tabpfn_challenger
	$(PY) -m src.models.backtest

test: setup
	$(PY) -m pytest -q
