CC ?= cc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

all: libfibembed.a example

libfibembed.a: fibembed.o
	ar rcs $@ $^

fibembed.o: fibembed.c fibembed.h fibembed_weights.h
	$(CC) $(CFLAGS) -c fibembed.c -o $@

example: example.c libfibembed.a
	$(CC) $(CFLAGS) example.c -L. -lfibembed $(LDFLAGS) -o $@

clean:
	rm -f *.o libfibembed.a example

.PHONY: all clean
