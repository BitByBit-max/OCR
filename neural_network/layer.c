#include "layer.h"


layer new_layer(int number_of_neurons)
{
	layer lay;
	lay.num_neu = -1;
	lay.neu = (struct neuron_t *) malloc(number_of_neurons * sizeof(struct neuron_t));
	return lay;
}
