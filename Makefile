run:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python Restricted_Boltzmann_Machine.py

test:
	python -m unittest discover -v
