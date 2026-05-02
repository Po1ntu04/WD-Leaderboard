PYTHON ?= python
RESULTS_DIR ?= platform/results
HOST ?= 0.0.0.0
PORT ?= 8050
FIGURES_DIR ?= $(RESULTS_DIR)/figures
FONT_PATH ?=

.PHONY: help install dashboard stop-dashboard test export-figures demo-session workspace-check

help:
	@echo "WD-Leaderboard developer commands"
	@echo "  make install          Install Python dependencies"
	@echo "  make dashboard        Run the Dash frontend on HOST=$(HOST) PORT=$(PORT)"
	@echo "  make stop-dashboard   Stop process listening on PORT=$(PORT)"
	@echo "  make test             Run pytest"
	@echo "  make export-figures   Export PPT-ready PNG figures"
	@echo "  make demo-session     Regenerate the small dashboard_sample results"
	@echo "  make workspace-check  Print workspace health and active dashboard process"

install:
	$(PYTHON) -m pip install -r requirements.txt

dashboard:
	$(PYTHON) app/leaderboard.py --results-dir $(RESULTS_DIR) --host $(HOST) --port $(PORT)

stop-dashboard:
	@lsof -ti:$(PORT) | xargs -r kill || true
	@echo "Stopped dashboard process on port $(PORT) if one existed."

test:
	$(PYTHON) -m pytest tests -q

export-figures:
	$(PYTHON) app/export_figures.py --results-dir $(RESULTS_DIR) --out-dir $(FIGURES_DIR) $(if $(FONT_PATH),--font-path $(FONT_PATH),)

demo-session:
	$(PYTHON) app/session.py \
		--prediction-dir test_assets/dashboard_sample/submissions \
		--raw test_assets/dashboard_sample/raw.txt \
		--gold test_assets/dashboard_sample/gold.txt \
		--manifest test_assets/dashboard_sample/manifest.csv \
		--results-dir platform/results/dashboard_sample

workspace-check:
	$(PYTHON) scripts/workspace_check.py --results-dir $(RESULTS_DIR) --port $(PORT)
