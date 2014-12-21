/** \file syntactic.h
 * \brief This is the header file containing all the classes and user variables.
 * 
 * This file contains the following classes:
 * class VVNeuron
 * class VVLayer
 * class VVNetwork
 * class VVTSNetwork
 */

#include<string>
#include<vector>
#include<iostream>
#include<stdarg.h>
#include<fstream>
#include "transfer_functions.h"
#include"randoom.h"
using namespace std;


/** \brief class VVNeuron is the most fundamental unit of the network
 *
 * This class is a friend of class VVLayer.
 * It contains 3 protected variables:
 */

class VVNeuron					
{
	friend class VVLayer;
	protected:
		/** \brief vvstate is the variable containing the state of the neuron 
		  *
		  */

	double vvstate;
	
	/** double accumulate is the variable containing the value 
	 *  which is passed to the transfer function to obtain the state.
	 */		
	double accumulate;

	/** double vvbias_weight is the weight of the biased input if biasing is enabled.
	 * 
	 */

	double vvbias_weight;
	public:

/** \brief This is the constructor.
 *
 * It initializes vvstate, accumulate and vvbias_weight to 0.
 */
	
	VVNeuron() 				
	{				
		 	
		vvstate=0;	
	
		accumulate=0;
		
		vvbias_weight=0;
	}

};

/** \brief class VVLayer is formed from a collection of neurons. Many layers form a network.
 * The class contains a pointer for pointing to an array of neurons. It contains all the details
 * which makes the layer complete.
 */
class VVLayer 
{

	protected:
		/** This is the pointer, to an array of neurons required in the layer*/
		VVNeuron * perceptrons;
		/** vvneuronnumbers is a variable that contains how many neurons are present in the layer. */
		long int vvneuronnumbers;
	public:
		/** \brief Due to the particular implementation of this class, vvlayerinit acts as a constructor which can be called.
		  *
		  * This function creates the neurons dynamicaly. It should not be called explicitly by the
		  * programmer unless he is creating a custom network or knows what he is doing.
		  */
		void vvlayerinit(long int a)	
		{
			vvneuronnumbers=a;
			perceptrons = new VVNeuron [a];
		}
		/** \brief Due to the particular implementation of this class, vvlayeruninit acts as a destructor which can be called.
		  *
		  * This function removes the neurons and deletes the pointer. It should not be called explicitly by the
		  * programmer unless he is creating a custom network or knows what he is doing.
		  */

		void vvlayeruninit()
		{
			
			delete [] perceptrons;
		}

		/** \brief Returns number of Neurons 
		 *
		 * This function returns the number of neurons present in the layer as long int.
		 */
		long int vvlayer_number_neuron()
		{
			return vvneuronnumbers;
		}
		
		/** \brief Sets biasing weight for a neuron
		 *
		 * This function sets the <index>th neuron's biasing weight with the value <val> in the current layer.
		 */
		void vvlayer_bias_perceprton(int index,double val)
		{
			perceptrons[index].vvbias_weight=val;
		}

		/** \brief Sets the state for a neuron
		 *
		 * This function sets the <index>th neuron's state with the value <val> in the current layer.
		 */
		void vvset_perceptron(int index,double val)
		{
			perceptrons[index].vvstate=val;
		}

		/** \brief Gets state of a neuron
		 *
		 * This function gets the <index>th neuron's state of the current layer, and returns it as a double.
		 */
		double vvget_perceptron(int index)
		{
			return(perceptrons[index].vvstate);
		}
		
		/** \brief Gets biasing weight of a neuron
		 *
		 * This function gets the <index>th neuron's biasing weight of the current layer, and returns it as a double.
		 */
		double vvbiased(int index)
		{
			return (perceptrons[index].vvbias_weight);
		}
		
		/** \brief Sets accumulated input of a neuron
		 *
		 * This function sets the <index>th neuron's accumulated input(input just before it passes
		 * into the transfer functions to obtain the state) to the value <S>, of the current layer.
		 */
		void vvset_perceptron_accumulator(int index,double S)
		{
			perceptrons[index].accumulate=S;
		}
		
		/** \brief Gets accumulated input of a neuron
		 *
		 * This function gets the <index>th neuron's accumulated input of the current layer, 
		 * and returns it as a double.
		 */
		double vvget_perceptron_accumulator(int index)
		{
			return perceptrons[index].accumulate;
		}



};

/** \brief This class is a predefined network using the FeedForward Backpropagation algorithm.
 *
 * class VVNetwork, is a template for the FeedForward Backpropagation algortihm. It supports networks, having
 * any number of layers. The network supports different transferfunctions.
 */
class VVNetwork
{
	protected:
		/** \brief This is a pointer to create any number of layers 
		 *
		 * The network consists of parallel layers with interconnections between each layer.
	         * Input is fed through the branches of various layers until it reaches the output layer.
		 */
		VVLayer * layers;
		/** \brief This is a very important concept in Neural Networks.
		 *
		 * The weights tell a neuron A, recieving input from neuron B, just how much importance,
		 * neuron A, must give to the input it recieves from neuron B. This is done by multiplying
		 * the input recieved with the weight.<br>
		 * The Three dimensional weights indicate:<br>
		 * The 1st dimension : It indicates the connection between the nth and (n+1)th layer.<br>
		 * The 2nd dimension : It indicates the ith neuron in the nth layer.
		 * The 3rd dimension : It indicates the jth neuron in the (n+1)th layer.
		 */
		double *** weights;
		/** This variable holds the number of layers in the network. */
		int no_of_layers;
		/** The meaning of this variable is self-evident */
		double learning_rate;
		/** The meaning of this variable is self-evident */	
		double momentum;
		/** Matrices to hold the input and output binary sets */
		double **inputs;
		/** Matrices to hold the input and output binary sets */
		double *outputs;
		/** Array which holds the number of neurons in each layer */
		vector <int> nonperlay;
		/** The meaning of this variable is self-evident */
		int no_training_records;
		/** The meaning of this variable is self-evident */
		int no_output_records;
		/** Internal variable, for neural network processing. */
		double **deli;
		/** A copy of the weight matrix for use in intermediate calculations*/
		double ***xerox_weights;
		/** The meaning of this variable is self-evident */
		double global_error;
		/** Annealing is under development. This variable is used for that purpose. Do not use. */
		double annealing_threshold;
		/** Iterational variables used for internal purposes */
		int iter;
		/** Iterational variables used for internal purposes */		
		int itercnt;
		/** The meaning of this variable is self-evident */
		double *error;
		/** Depreciated variable */
		int progbar;
		/** Internal function to initiate Feed Forward process */
		void VVFeedForward(int&);
		/** Function to initiate Back Propagation process */
		void VVBackPropagate(int&);
	public:
		
		/** This function pointer is for indicating the type of transfer functions */
		double(*vvtransfer_function)(double);
		/** This function pointer is for indicating the type of derivative of transfer functions */
		double(*vvtransfer_function_derivative)(double);
		/** This function extracts input and stores it in memory from the specified file. */
		void VVreadinputfrom(char * );
		/** This function extracts the actual output and stores it in memory from the specfied file. */
		void VVreadoutputfrom(char *);
		/** This function enables autobiasing for all non-input neurons. */
		void vvautobias();
		/** \brief Trains the network with the specified data.
		 *
		 * The function is used to train the network for a single iteration. It calls
		 * VVFeedForward(int&) and VVBackPropagate(int&).
		 */
		void VVnetwork_train();
		/** This function tests the network with the specified testfile. */
		vector<double> VVtest_network(char[]);
		/** \brief This function saves the network.
		 *
		 * This function saves all the vital characterestics of a network such as
		 * the number of layers, the number of neurons per layer, the weights, etc.
		 * This is stored in a binary format for ease of saving and loading.
		 * Hence do not attempt to a saved network, or it may end up curropted.
		*/
		int VVsave_network(char[]);
		/** This function loads all the data saved from the network by the above function */
		int VVload_network(char[]);
		/** This is a debugging function to show all the weights. */
		void show_weights();
		/** Depreciated function */
		void VVProgressBar(int);
		/** This function is in research stages. Do not use. */
		void VVSimulatedAnnealing(double);
		VVNetwork(){}
		/** \brief Constructor, uses ellipsis.
		 *
		 * The constructor follows the follwing format:<br>
		 *	<ul>
		 *	The first argument is the number of layers(int). <br>
		 *	The second,third...(n-3)th arguments indicate the number of neurons(int) for concecutive layers.<br>
		 *	The third last argument is the learning rate(double). <br>
		 * 	The second last argument is the momentum value(double). <br>
		 * 	The last argument is the number of iterations to be executed. <br>
		 *   	</ul>
		 */
		VVNetwork(int,...);
		/** The destructor, it frees all the dynamically allocated memory */
		~VVNetwork()
		{
			
			for(int i=0;i<no_of_layers;i++)
			layers[i].vvlayeruninit();
			delete[] layers;
			delete[] weights;
			delete[] inputs;
			delete[] outputs;
			delete[] xerox_weights;
			delete[] deli;
			delete error;
			vvtransfer_function=NULL;
			vvtransfer_function_derivative=NULL;
			
		}



};

class TSVVNetwork
{
	protected:
		VVLayer * layers;
		double *** weights;
		int no_of_layers;
		double learning_rate,momentum;
		double **inputs,*outputs;
		vector <int> nonperlay;
		int no_training_records,no_output_records;
		double **deli;
		double ***xerox_weights;
		double global_error,annealing_threshold;
		int iter,itercnt;
		double *error;
		int progbar;
		void VVFeedForward(int&);
		void VVBackPropagate(int&);
	public:
		double(*vvtransfer_function)(double);
		double(*vvtransfer_function_derivative)(double);
		void VVreadinputfrom(char * );
		void VVreadoutputfrom(char *);
		void vvautobias();
		void VVnetwork_train();
		vector<double> VVtest_network(char[]);
		void VVsave_network(char[]);
		int VVload_network(char[]);
		void show_weights();
		double*  VVGetHiddenLayerState(int);
		void VVProgressBar(int);
		void VVSimulatedAnnealing(double);
		TSVVNetwork(){}
		TSVVNetwork(int,...);
		~TSVVNetwork()
		{
			for(int i=0;i<no_of_layers;i++)
			layers[i].vvlayeruninit();
			delete[] layers;
			delete[] weights;
			delete[] inputs;
			delete[] outputs;
			delete[] xerox_weights;
			delete[] deli;
			vvtransfer_function=NULL;
			vvtransfer_function_derivative=NULL;
			
		}

	
};
