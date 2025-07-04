.SILENT: run

install:
	pip install -r requirements.txt

run/1:
	python3 1_malha_aberta.py

run/2:
	python3 2_malha_fechada_control.py

run/3:
	python3 3_malha_fechada_control_ref.py

run/4:
	python3 4_malha_fechada_control_ref_estim.py