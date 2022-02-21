targets := int32 int64 float32 float64

all: $(targets)
	numba add.py merge *.o

run:
	python -c "import my_module; print(dir(my_module))"

$(targets):
	numba add.py emit-obj -f add -n add_$@ -s "$@($@, $@)" -o add_$@.o

clean:
	rm -f *.o *.o.pickle *.so