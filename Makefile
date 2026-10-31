PY ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
BIN = $(VENV)/Scripts
else
BIN = $(VENV)/bin
endif
IMAGE ?= riverbend-assistant

.PHONY: setup corpus ingest run test eval docker-build docker-run

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt

corpus:
	$(BIN)/python scripts/build_corpus.py

ingest:
	$(BIN)/python -m app.ingest --reset

run:
	$(BIN)/streamlit run ui/streamlit_app.py

test:
	$(BIN)/python -m pytest -q

eval:
	$(BIN)/python -m eval.run_eval

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --env-file .env -p 8080:8080 $(IMAGE)
