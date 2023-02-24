#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <stdlib.h>
#include <err.h>
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "pixel_operations.h"
#include <stdio.h>
#include <sys/stat.h>
#include <math.h>
#include <dirent.h>

void init_sdl2();
SDL_Surface* load_imag(char *path);
double sigmoid(double x);
double dSigmoid(double x);
double init_weight();
void init(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,
		double* hiddenWeights,double* outputLayerBias,
		double* hiddenLayerBias,
		double* training_outputs);
void shuffle(int *array, size_t n);
int feedforward(int numHiddenNodes, int numOutputs,int numInputs,
		double* inputLayer,double* outputWeights,
		double* hiddenWeights, double* outputLayerBias,
		double* hiddenLayerBias,double* hiddenLayer,double* outputLayer);
void train_data(double* res, int digit, int k, int i, int j);
void train(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,double* hiddenWeights,
		double* outputLayerBias,double* hiddenLayerBias,
		double* hiddenLayer,double* outputLayer,
		double* training_outputs);
void save(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,double* hiddenWeights,
		double* outputLayerBias,double* hiddenLayerBias);
void clean_str(char* s);
double run_strtod (char* input);
void load(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,double* hiddenWeights,
		double* outputLayerBias,double* hiddenLayerBias);
#endif