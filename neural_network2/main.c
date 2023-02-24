#include <stdlib.h>
#include <err.h>
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "pixel_operations.h"
#include "neural_network.h"
#include <stdio.h>
#include <sys/stat.h>
#include <math.h>
#include <dirent.h>

typedef struct Neural_network
{
	int numInputs;
	int numHiddenNodes;
	int numOutputs;

	double* hiddenLayer;
	double* outputLayer;
	double* hiddenLayerBias;
	double* outputLayerBias;
	double* hiddenWeights;
	double* outputWeights;
	double* training_outputs;
	double* inputLayer;
} Neural_network;

int main(int argc,char *argv[])
{
    if (argc != 3)
	{
		printf("wrong number of arguments\n");
		exit(EXIT_FAILURE);
	}
	
	Neural_network nn ={
	.numInputs = 784,
	.numHiddenNodes = 20,
	.numOutputs = 10,

	.hiddenLayer = malloc(sizeof(double)*20),
	.outputLayer = malloc(sizeof(double)*10),
	.hiddenLayerBias = malloc(sizeof(double)*20),
	.outputLayerBias = malloc(sizeof(double)*10),
	.hiddenWeights = malloc(sizeof(double)*784*20),
	.outputWeights = malloc(sizeof(double)*20*10),
	.training_outputs = malloc(sizeof(double)*10*10),
	.inputLayer = malloc(sizeof(double)*784),
	};

	init(nn.numHiddenNodes,nn.numOutputs,nn.numInputs,
	nn.outputWeights, nn.hiddenWeights,nn.outputLayerBias,
	nn.hiddenLayerBias,nn.training_outputs);

	if (argv[2][0]=='1') //then we train
	{
		train(nn.numHiddenNodes, nn.numOutputs, nn.numInputs,
		nn.outputWeights, nn.hiddenWeights,
		nn.outputLayerBias, nn.hiddenLayerBias,
		nn.hiddenLayer, nn.outputLayer,
		nn.training_outputs);
		save(nn.numHiddenNodes, nn.numOutputs, nn.numInputs,
		nn.outputWeights, nn.hiddenWeights,
		nn.outputLayerBias, nn.hiddenLayerBias);
		return 0;
	}
	else
	{
		load(nn.numHiddenNodes, nn.numOutputs, nn.numInputs,
		nn.outputWeights, nn.hiddenWeights,
		nn.outputLayerBias, nn.hiddenLayerBias);
		SDL_Surface* image = load_imag(argv[1]);
    	double res [784];
		for(size_t y = 0; y < 28; y++){
			for(size_t x = 0; x < 28; x++){
				uint32_t pixel = get_pixel(image, x, y);
				uint8_t r,g,b;
				SDL_GetRGB(pixel, image->format, &r, &g, &b);
				res[y*28 + x] = r != 0;
				}
			}
		int m = feedforward(nn.numHiddenNodes, nn.numOutputs, nn.numInputs,
					res, nn.outputWeights,
					nn.hiddenWeights, nn.outputLayerBias,
					nn.hiddenLayerBias, nn.hiddenLayer,nn.outputLayer);
		printf("%d",m);
		return m;
	}
}
