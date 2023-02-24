CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -O1 
LDLIBS = `pkg-config --cflags --libs sdl` -lSDL_image -lm -ldl 

main: neural_network.o pixel_operations.o main.o
	$(CC) -o main main.o neural_network.o pixel_operations.o -I/usr/local/include -L/usr/local/lib -lSDL $(LDLIBS)

main.o:
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c
	$(CC) $(CFLAGS) -c neural_network.c

pixel_operations.o: pixel_operations.c
	$(CC) $(CFLAGS) -c pixel_operations.c

clean:
	${RM} -f *.o
	${RM} -f *.d
	${RM} -f *.gch
	${RM} pixel_operations
	${RM} neural_network
	${RM} main
