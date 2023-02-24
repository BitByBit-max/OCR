#include "layer.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


layer *layers = NULL;
int num_layers = 3; //1 input + 1 hidden + 1 output
int *num_neurons;
float alpha = 0.8;
float *cost;
float full_cost;
float **input;
float **desired_outputs;
int num_training_ex;
int n=1;

int initialize_weights(void)
{
    int i,j,k;

    if(layers == NULL)
    {
        printf("No layers in Neural Network...\n");
        return 1;
    }

    printf("Initializing weights...\n");

    for(i=0;i<num_layers-1;i++)
    {
        
        for(j=0;j<num_neurons[i];j++)
        {
            neuron n = layers[i].neu[j];
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Initialize Output Weights for each neuron
                n.out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, n.out_weights[k]);
                n.dw[k] = 0.0;
            }

            if(i>0) 
            {
                layers[i].neu[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }   
    printf("\n");
    
    for (j=0; j<num_neurons[num_layers-1]; j++)
    {
        layers[num_layers-1].neu[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return 0;
}

// Create Neural Network Architecture
int create_architecture(void)//DONE
{
    int i=0,j=0;
    layers = (layer*) malloc(num_layers * sizeof(layer));

    for(i=0;i<num_layers;i++)
    {
        layers[i] = new_layer(num_neurons[i]);      
        layers[i].num_neu = num_neurons[i];
        printf("Created Layer: %d\n", i+1);
        printf("Number of Neurons in Layer %d: %d\n", i+1,layers[i].num_neu);

        for(j=0;j<num_neurons[i];j++)
        {
            if(i < (num_layers-1)) 
            {
                layers[i].neu[j] = new_neuron(num_neurons[i+1]);
            }

            printf("Neuron %d in Layer %d created\n",j+1,i+1);  
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if(initialize_weights() != 0)
    {
        printf("Error Initilizing weights...\n");
        return 1;
    }

    return 0;
}

int init()
{
    if(create_architecture() != 0)
    {
        printf("Error in creating architecture...\n");
        return 1;
    }

    printf("Neural Network Created Successfully...\n\n");
    return 0;
}

//Get Inputs
void  get_inputs() //I'll code that differenty for the real one obviously but it makes things easier to test for the xor function
{
    int i,j;

        for(i=0;i<num_training_ex;i++)
        {
            printf("Enter the Inputs[%d]:\n",i);

            for(j=0;j<num_neurons[0];j++)
            {
                scanf("%f",&input[i][j]);
                
            }
            printf("\n");
        }
}

//Get Labels
void get_desired_outputs() //I'll hardcode that for the real one as well
{
    int i,j;
    
    for(i=0;i<num_training_ex;i++)
    {
        for(j=0;j<num_neurons[num_layers-1];j++)
        {
            printf("Enter the Desired Outputs[%d]: \n",i);
            scanf("%f",&desired_outputs[i][j]);
            printf("\n");
        }
    }
}

// Feed inputs to input layer
void feed_input(int i) //DONE
{
    for(int j=0;j<num_neurons[0];j++)
    {
        layers[0].neu[j].actv = input[i][j];
        printf("Input: %f\n",layers[0].neu[j].actv);
    }
}


void update_weights(void) // DONE
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                neuron n = layers[i].neu[j];
                n.out_weights[k] -= (alpha * n.dw[k]);
            }
            
            // Update Bias
            layers[i].neu[j].bias -= (alpha * layers[i].neu[j].dbias);
        }
    }   
}

void forward_prop(void)
{
    int i,j,k;

    for(i=1;i<num_layers;i++)
    {   
        for(j=0;j<num_neurons[i];j++)
        {
            layers[i].neu[j].z = layers[i].neu[j].bias;

            for(k=0;k<num_neurons[i-1];k++)
            {
                neuron prev = layers[i-1].neu[k];
                layers[i].neu[j].z  += ((prev.out_weights[j])* (prev.actv));
            }

            // Relu Activation Function for Hidden Layers
            if(i < num_layers-1)
            {
                if((layers[i].neu[j].z) < 0)
                {
                    layers[i].neu[j].actv = 0;
                }

                else
                {
                    layers[i].neu[j].actv = layers[i].neu[j].z;
                }
            }
            
            // Sigmoid Activation function for Output Layer
            else
            {
                layers[i].neu[j].actv = 1/(1+exp(-layers[i].neu[j].z));
                printf("Output: %d\n", (int)round(layers[i].neu[j].actv));
                printf("\n");
            }
        }
    }
}

// Compute Total Cost
void compute_cost(int i) 
{
    int j;
    float tmpcost=0;
    float totcost=0;

    for(j=0;j<num_neurons[num_layers-1];j++)
    {
        tmpcost = desired_outputs[i][j] - layers[num_layers-1].neu[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        totcost += cost[j];
    }   

    full_cost = (full_cost + totcost)/n;
    n++;
    //printf("Full Cost: %f\n",full_cost);
}

// Back Propogate Error
void back_prop(int p) //1 over 80
{
    int i,j,k;
    // Output Layer
    for(j=0;j<num_neurons[num_layers-1];j++)
    {           
        float o= desired_outputs[p][j];
        layers[num_layers-1].neu[j].dz = (layers[num_layers-1].neu[j].actv-o)* 
                                        (layers[num_layers-1].neu[j].actv) * 
                                        (1- layers[num_layers-1].neu[j].actv);

        for(k=0;k<num_neurons[num_layers-2];k++)
        {   
            layers[num_layers-2].neu[k].dw[j]=layers[num_layers-1].neu[j].dz*
                                            layers[num_layers-2].neu[k].actv;
            float w = layers[num_layers-2].neu[k].out_weights[j];//this is only to fit the 80 columns thing
            layers[num_layers-2].neu[k].dactv = w * 
                                            layers[num_layers-1].neu[j].dz;
        }
            
        layers[num_layers-1].neu[j].dbias = layers[num_layers-1].neu[j].dz;           
    }
    
    // Hidden Layers but we have only one it's the one at index 1
    for(j=0;j<num_neurons[1];j++)
    {
        if(layers[1].neu[j].z >= 0)
        {
            layers[1].neu[j].dz = layers[1].neu[j].dactv;            
        }
        else
        {
            layers[1].neu[j].dz = 0;
        }

        for(k=0;k<num_neurons[0];k++)
        {
            neuron n = layers[0].neu[k];
            n.dw[j] = layers[1].neu[j].dz * n.actv;   
        }
        layers[1].neu[j].dbias = layers[1].neu[j].dz;
    }
}

// Train Neural Network
void train_neural_net(void) //DONE
{
    int i;
    int it=0; //nbr of iterations

    // Gradient Descent
    for(it=0;it<20000;it++)
    {
        for(i=0;i<num_training_ex;i++)
        {
            feed_input(i);
            forward_prop();
            compute_cost(i);
            back_prop(i);
            update_weights();
        }
    }
}

// Test the trained network
void test_nn(void) 
{
    int i;
    while(1)
    {
        printf("Enter input to test:\n");

        for(i=0;i<num_neurons[0];i++)
        {
            scanf("%f",&layers[0].neu[i].actv);
        }
        forward_prop();
    }
}

int main(void) //1 over 80
{
    int i;
    num_neurons = (int*) malloc(num_layers * sizeof(int));
    memset(num_neurons,0,num_layers *sizeof(int));

    // number of neurons per layer 
    num_neurons[0] = 2;
    num_neurons[1] = 2;
    num_neurons[2] = 1;

    // Initialize the neural network module
    if(init()!= 0)
    {
        printf("Error in Initialization...\n");
        exit(0);
    }

    printf("Enter the number of training examples: \n");
    scanf("%d",&num_training_ex);
    printf("\n");

    input = (float**) malloc(num_training_ex * sizeof(float*));
    for(i=0;i<num_training_ex;i++)
    {
        input[i] = (float*)malloc(num_neurons[0] * sizeof(float));
    }

    desired_outputs = (float**) malloc(num_training_ex* sizeof(float*));
    for(i=0;i<num_training_ex;i++)
    {
        int nnbr =num_neurons[num_layers-1];
        desired_outputs[i] = (float*)malloc(nnbr * sizeof(float));
    }

    cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));
    memset(cost,0,num_neurons[num_layers-1]*sizeof(float));

    // Get Training Examples
    get_inputs();

    // Get Output Labels
    get_desired_outputs();

    train_neural_net();
    test_nn();

    return 0;
}
