targets := int32 int64 float32 float64

all: $(targets)
	python ../../numba-aot.py add.py merge *.o -o add.so

run:
	python -c "import lib; print(dir(lib))"

$(targets):
	python ../../numba-aot.py add.py emit-obj -f add -n add_$@ -s "$@($@, $@)" -o add_$@.o

clean:
	rm -f *.o *.pickle *.so