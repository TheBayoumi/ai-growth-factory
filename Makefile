.PHONY: test preflight doctor modal-deploy modal-canary

test:
	python -m unittest discover -s tests -v

preflight:
	python scripts/repository_preflight.py

doctor:
	python -m factory doctor

modal-deploy:
	modal deploy cloud/modal_app.py

modal-canary:
	modal run cloud/modal_app.py --canary
