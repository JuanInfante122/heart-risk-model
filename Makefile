SHELL := /bin/bash

.PHONY: setup install run clean

setup:
	@echo "Creando entorno virtual..."
	python -m venv venv
	@echo "Entorno virtual 'venv' creado."

install:
	@echo "Instalando dependencias..."
	@source venv/bin/activate && pip install -r requirements/requirements.txt
	@echo "Dependencias instaladas."

run:
	@echo "Iniciando la aplicacion..."
	@source venv/bin/activate && streamlit run app/app.py

clean:
	@echo "Limpiando..."
	@rm -rf venv
	@echo "Entorno virtual eliminado."
