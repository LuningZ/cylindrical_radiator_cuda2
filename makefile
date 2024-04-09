# Compilers
CC= g++
NVCC= nvcc

# Flags
CFLAGS= -W -Wall
NVCCFLAGS= -g -G --use_fast_math

INCPATH= /usr/include/

all: cpu float double

cpu: cpu.c
	$(CC) -o cpu cpu.c $(CFLAGS)
float: float.cu
	$(NVCC) -o float float.cu $(NVCCFLAGS) -I$(INCPATH)
double: double.cu
	$(NVCC) -o double double.cu $(NVCCFLAGS) -I$(INCPATH)


clean:
	rm -f cpu float double
