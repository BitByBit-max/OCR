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
void init_sdl2()
{
    if(SDL_Init(SDL_INIT_VIDEO) == -1)
        errx(1,"Could not initialize SDL: %s. \n", SDL_GetError());
}

SDL_Surface* load_imag(char *path)
{
    SDL_Surface *img;
    img = IMG_Load(path);
    if (!img)
        errx(3, "can't load %s: %s", path, IMG_GetError());
    return img;
}

//TOOLS

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
double init_weight() { return ((double)rand())/((double)RAND_MAX); }


void init(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,
		double* hiddenWeights,double* outputLayerBias,
		double* hiddenLayerBias,
		double* training_outputs)
{

	//init training_outputs
	for(int x=0; x<numOutputs; x++)
	{
		for(int y=0; y<numOutputs; y++)
		{
			if(y == x)
			{
				training_outputs[x*numOutputs+y] = 1.0f;
			}
			else
			{
				training_outputs[x*numOutputs+y] = 0.0f;
			}
		}
	}
	//Init Weights
	for (int i=0; i<numInputs; i++)
	{
		for (int j=0; j<numHiddenNodes; j++)
		{
			hiddenWeights[i*numHiddenNodes+j] = init_weight();
		}
	}
	for (int i=0; i<numHiddenNodes; i++)
	{
		hiddenLayerBias[i] = init_weight();
		for (int j=0; j<numOutputs; j++)
		{
			outputWeights[i*numOutputs+j] = init_weight();
		}
	}
	for (int i=0; i<numOutputs; i++)
	{
		outputLayerBias[i] = init_weight();
	}
}

void shuffle(int *array, size_t n)
{
	if (n > 1)
	{
		size_t i;
		for (i = 0; i < n - 1; i++)
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

//FEEDFORWARD
int feedforward(int numHiddenNodes, int numOutputs,int numInputs,
		double* inputLayer,double* outputWeights,
		double* hiddenWeights, double* outputLayerBias,
		double* hiddenLayerBias,double* hiddenLayer,double* outputLayer)
{
	//Compute Hidden layer activation
	for (int j=0; j<numHiddenNodes; j++)
	{
		double activation=hiddenLayerBias[j];
		for (int k=0; k<numInputs; k++)
		{
			activation += inputLayer[k]*hiddenWeights[k*numHiddenNodes+j];
		}
		hiddenLayer[j] = sigmoid(activation);
	}

	// OUTPUT LAYER - SOFTMAX ACTIVATION
	double esum = 0.0;
	for (int i=0;i<numOutputs;i++)
	{
		double sum = 0.0;
		for (int j=0;j<numHiddenNodes;j++)
		{
			sum += hiddenLayer[j]*outputWeights[j*numOutputs+i];
		}
		outputLayer[i] = sum+outputLayerBias[i];
		esum += exp(outputLayer[i]);
	}

	// SOFTMAX FUNCTION
	for (int i=0;i<numOutputs;i++)
	{
		outputLayer[i] = exp(outputLayer[i])/ esum;
	}
	int imax=0;
	for(int i=1; i<numOutputs;i++)
	{
		if(outputLayer[imax]<outputLayer[i])
		{
			imax = i;
		}
	}
	return imax;
}
void train_data(double* res, int digit, int k, int i, int j)
{
	SDL_Surface* image;
	if(k==0)
	{
		char path[] = {'t','r','a','i','n','i','n','g','_','n','u','m'
			,'b','e','r','/',48+(char)digit,'/',48+(char)i,48+(char)j,'.'
				,'b','m','p',0};
		image = load_imag(path);
	}
	else
	{
		char path2[] = {'t','r','a','i','n','i','n','g','_','n','u','m'
			,'b','e','r','/',48+(char)digit,'/',48+(char)k,48+(char)i,48+(char)j,'.'
				,'b','m','p',0};
		image = load_imag(path2);
	}

	if(!image)
	{
		printf("not a good path");
	}
	else
	{
		for(size_t y = 0; y < 28; y++)
		{
			for(size_t x = 0; x < 28; x++)
			{
				uint32_t pixel = get_pixel(image, x, y);
				uint8_t r,g,b;
				SDL_GetRGB(pixel, image->format, &r, &g, &b);
				if(r != 0)
				{
					res[y*28 + x] = 1;
				}
				else
				{
					res[y*28 +x] = 0;
				}
			}
		}
	}
}
//TRAIN
void train(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,double* hiddenWeights,
		double* outputLayerBias,double* hiddenLayerBias,
		double* hiddenLayer,double* outputLayer,
		double* training_outputs)
{
	//Init Training
	int numTrainingSets = numOutputs;
	int trainingSetOrder[] = {0,1,2,3,4,5,6,7,8,9};
	int nbfiles[] = {45,20,20,20,20,20,20,20,20,20};
	int nbcount[] = {0,0,0,0,0,0,0,0,0,0};
	const double lr = 0.1f;
	double* res = malloc(sizeof(double)*784); //contains the locations of the training digits converted in arrays
	//Iterate through the entire training for a nb of epochs(10000 here)
		for (int n=0; n <2000;n++)
		{
		// As per SGD, shuffle the order of the training set
		shuffle(trainingSetOrder, numTrainingSets);

		// Cycle through each of the training set elements
		for (int x=0; x<numTrainingSets; x++)
		{
			int i = trainingSetOrder[x];
			for(; nbcount[i]<nbfiles[i]; nbcount[i]++)
			{
				//Load Data
				train_data(res,i,nbcount[i]/100,
					(nbcount[i]/10)%10,nbcount[i]%10);

				//FEEDFORWARD

				int max = feedforward(numHiddenNodes, numOutputs,
						numInputs,
						res,
						outputWeights,
						hiddenWeights,
						outputLayerBias,
						hiddenLayerBias,
						hiddenLayer,
						outputLayer);
				printf("\nOutput: %4d Expected Output: %4d  debug: %4g \n",max,i, outputLayer[max]);

				// Backprop
				// Compute change in output weights
				double deltaOutput[numOutputs];
				for (int j=0; j<numOutputs; j++)
				{
					double errorOutput = training_outputs[i*numOutputs+j]-outputLayer[j];
					deltaOutput[j] = errorOutput;
				}

				// Compute change in hidden weights
				double deltaHidden[numHiddenNodes];
				for (int j=0; j<numHiddenNodes; j++)
				{
					double errorHidden = 0.0f;
					for(int k=0; k<numOutputs; k++)
					{
						errorHidden+=deltaOutput[k]*outputWeights[j*numOutputs+k];
					}
					deltaHidden[j] = errorHidden*dSigmoid(hiddenLayer[j]);
				}

				// Apply change in output
				for (int j=0; j<numOutputs; j++)
				{
					outputLayerBias[j] += deltaOutput[j]*lr;
					for (int k=0; k<numHiddenNodes; k++)
					{
						outputWeights[k*numOutputs+j]+=hiddenLayer[k]*deltaOutput[j]*lr;
					}
				}

				// Apply change in hidden
				for (int j=0; j<numHiddenNodes; j++)
				{
					hiddenLayerBias[j] += deltaHidden[j]*lr;
					for(int k=0; k<numInputs; k++)
					{
						hiddenWeights[k*numHiddenNodes+j]+=res[k]*deltaHidden[j]*lr;
					}
				}
			}
		}
		for(int z =0; z<10;z++)
		{
			nbcount[z] = 0;
		}
	}
}

void save(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,double* hiddenWeights,
		double* outputLayerBias,double* hiddenLayerBias)
{
	/* File pointer to hold reference to our file */
	FILE * fPtr;
	// Open file in w (write) mode.
	fPtr = fopen("weight_save", "w");
	//allocate memory for the conversion (double to string)
	char str[21];


	/* fopen() return NULL if last operation was unsuccessful */
	if(fPtr == NULL)
	{
		/* File not created hence exit */
		printf("Unable to create file.\n");
		exit(EXIT_FAILURE);
	}
	//Write In the file
	fputs("|Hidden Weights|\n", fPtr);
	for (int j=0; j<numHiddenNodes; j++)
	{
		for(int k=0; k<numInputs; k++)
		{
			sprintf(str, "%f", hiddenWeights[k*numHiddenNodes+j]);
			fputs(str, fPtr);
			fputs("\n", fPtr);
		}
	}
	fputs("|Hidden Biases|\n", fPtr);
	for (int j=0; j<numHiddenNodes; j++)
	{
		sprintf(str, "%f", hiddenLayerBias[j]);
		fputs(str, fPtr);
		fputs("\n", fPtr);
	}
	fputs("|Output Weights|\n", fPtr);
	for (int j=0; j<numOutputs; j++)
	{
		for (int k=0; k<numHiddenNodes; k++)
		{
			sprintf(str, "%f", outputWeights[k*numOutputs+j]);
			fputs(str,fPtr);
			fputs("\n", fPtr);
		}
	}
	fputs("|Output Biases|\n", fPtr);
	for (int j=0; j<numOutputs; j++)
	{
		sprintf(str, "%f", outputLayerBias[j]);
		fputs(str, fPtr);
		fputs("\n", fPtr);
	}

	/* Close file to save file data */
	fclose(fPtr);


	/* Success message */
	printf("File created and saved successfully. :)\n\n");
}
//FOR LOAD
void clean_str(char* s)
{
	while(*s != 0)
	{
		*s = 0;
		s++;
	}
}

//Convert string to double
double run_strtod (char* input)
{
	double output;
	char * end;

	output = strtod (input, & end);

	if (end == input)
	{
		printf (" is not a valid number.\n");
	}
	return output;
}

//LOAD
void load(int numHiddenNodes, int numOutputs,int numInputs,
		double* outputWeights,double* hiddenWeights,
		double* outputLayerBias,double* hiddenLayerBias)
{
	int MAX_LINE_LENGTH = 100;

	FILE * fPtr;
	// Open file in r (read) mode.
	fPtr = fopen("weight_save", "r");
	//allocate memory
	char str[21];
	char line[100];

	/* fopen() return NULL if last operation was unsuccessful */
	if(fPtr == NULL)
	{
		/* File not created hence exit */
		printf("Unable to load weight.\n");
		exit(EXIT_FAILURE);
	}
	//Read the file
	fgets(line, MAX_LINE_LENGTH, fPtr);
	int i = 0;
	for (int j=0; j<numHiddenNodes; j++)
	{
		for(int k=0; k<numInputs; k++)
		{
			fgets(line, MAX_LINE_LENGTH, fPtr);
			int m =0;
			i=0;
			while(line[i] != '\n' && line[i] != '|' && line[i]!=0)
			{
				str[m] = line[i];
				i++;
				m++;
			}
			hiddenWeights[k*numHiddenNodes+j] = run_strtod(&str[0]);
			clean_str(&str[0]);
		}
	}

	fgets(line, MAX_LINE_LENGTH, fPtr);
	for (int j=0; j<numHiddenNodes; j++)
	{
		i=0;
		int m=0;
		fgets(line, MAX_LINE_LENGTH, fPtr);
		while(line[i] != '\n' && line[i] != '|' && line[i]!=0)
		{
			str[m] = line[i];
			i++;
			m++;
		}
		hiddenLayerBias[j] = run_strtod(&str[0]);
		clean_str(&str[0]);
	}
	fgets(line, MAX_LINE_LENGTH, fPtr);
	i=0;
	for (int j=0; j<numOutputs; j++)
	{
		for (int k=0; k<numHiddenNodes; k++)
		{

			fgets(line, MAX_LINE_LENGTH, fPtr);
			i = 0;
			int m=0;
			while(line[i] != '\n' && line[i] != '|' && line[i]!=0)
			{
				str[m] = line[i];
				i++;
				m++;
			}
			outputWeights[k*numOutputs+j] = run_strtod(&str[0]);
			clean_str(&str[0]);
		}
	}
	fgets(line, MAX_LINE_LENGTH, fPtr);
	i=0;
	for (int j=0; j<numOutputs; j++)
	{
		int m=0;
		fgets(line, MAX_LINE_LENGTH, fPtr);
		i=0;
		while(line[i] != '\n' && line[i] != '|' && line[i]!=0)
		{
			str[m] = line[i];
			i++;
			m++;
		}
		outputLayerBias[j] = run_strtod(&str[0]);
		clean_str(&str[0]);
	}
	clean_str(&line[0]);
	/* Close file to save file data */
	fclose(fPtr);
	printf("loaded successfully. :)\n\n");
}

