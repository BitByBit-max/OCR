CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -O1
#LDLIBS = `pkg-config --cflags --libs sdl2` -lSDL2_image -lm //for image processing

main: main.c layer.o neuron.o
	$(CC) -o main main.c layer.o neuron.o -lm

layer.o: layer.c
	$(CC) $(CFLAGS) -c layer.c layer.h

neuron.o: neuron.c
	$(CC) $(CFLAGS) -c neuron.c neuron.h

clean:
	${RM} -f *.o
	${RM} -f *.d
	${RM} neuron
	${RM} layer
	${RM} main