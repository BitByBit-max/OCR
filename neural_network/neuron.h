#ifndef NEURON_H
#define NEURON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct neuron_t
{
	float actv;//output value
	float *out_weights;
	float bias;
	float z;//sum of weights with each neuron plus bias

	float dactv;
	float *dw;
	float dbias;
	float dz;


} neuron;

neuron new_neuron(int num_out_weights);

#endif