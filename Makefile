CC	= gcc
FLAGS	= -fopenacc -lcuda -fcf-protection=none 

all: test.ex cuda_base.ex

test.ex: Mat_add.o timer.o
	module unload soft/cuda; $(CC) $(FLAGS) Mat_add.o timer.o -o test.ex -lm 

cuda_base.ex: 
	module load soft/cuda; nvcc MatAddN.cu -o cuda_base.ex

.c.o:
	$(CC) $(FLAGS) $< -c -o $@

clean:
	rm -f *.o *~ *.ex
