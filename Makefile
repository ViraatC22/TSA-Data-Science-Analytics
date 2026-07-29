PYTHON ?= python3

.PHONY: verify compile test smoke
verify: compile test smoke

compile:
	$(PYTHON) -m compileall -q TSA_Data_Science_Analytics.py tsa_analysis.py scripts tests

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	$(PYTHON) scripts/smoke_streamlit.py
