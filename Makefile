PY ?= python3
.PHONY: install test demo preflight
install:
	$(PY) -m pip install -r requirements.txt
	cd frontend && npm install
test:
	$(PY) -m pytest -q
demo:
	cd frontend && npm install && npm run build
	$(PY) -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
preflight:
	$(PY) scripts/preflight_live.py
