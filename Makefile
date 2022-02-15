help:
	@echo make env
	@echo make install
	@echo make run

env:
	pipenv shell

install:
	pipenv install -requirements.txt

run:
	python3 main.py
