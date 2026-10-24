PYTHON ?= python

.PHONY: test test-all
# The fast subset: everything except the tests marked slow.
test:
	$(PYTHON) -m pytest -q -m "not slow"

# Every test, for a release or a change to frames.py or proxy.py.
test-all:
	$(PYTHON) -m pytest -q
