NPROC ?= 4

format:
	ruff check --fix && ruff format
