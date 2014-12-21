#include "syntactic.h"
void VVNetwork :: show_weights()
{

	cout << "Weights: " << endl;
	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{
				cout << weights[i][j][k] << endl;

			}
}

void VVNetwork :: vvautobias()
{
	randoom r1(1,12344*time(0));
	r1.generate();
	for(int i=1;i<no_of_layers;i++)
		for(int j=0;j<nonperlay[i];j++)
		{

			double val = r1.generate();
			if(val*r1.generate()<r1.generate()) val*=-1.0;
			layers[i].vvlayer_bias_perceprton(j,val);
		}
}
void VVNetwork :: VVProgressBar(int i=1)
{
	if(progbar)   
	{
		if(i%(iter/100)==0)
		{
			if(i/(iter/100)==99)
				cout<<'|'<<endl;				
			else
			cout<<'*'<<flush;
		}	
	}		
	else 
	{
		cout<<endl<<'|';
		progbar=1;
	}
}
void VVNetwork :: VVSimulatedAnnealing(double thres)
{
	annealing_threshold=thres;
}
VVNetwork :: VVNetwork(int a,...)
{

	va_list ap;
	va_start(ap, a);
	layers = new VVLayer[a];
	for(int i=0;i<a;i++)
		nonperlay.push_back(va_arg(ap, int));	
	for(int i=0;i<a;i++)
		layers[i].vvlayerinit(nonperlay[i]);
	no_of_layers=a;
	learning_rate=va_arg(ap,double);
	momentum=va_arg(ap,double);
	iter=va_arg(ap,int);
	randoom rn1(1,67435*time(0)+a);
	rn1.generate();
	progbar=0;
	weights=new double**[no_of_layers-1];
	xerox_weights=new double**[no_of_layers-1];

	for(int i=0;i<no_of_layers-1;i++)
	{
		weights[i]=new double*[nonperlay[i]];
		xerox_weights[i] = new double*[nonperlay[i]];
	}
	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			weights[i][j]= new double [nonperlay[i+1]];
			xerox_weights[i][j]= new double[nonperlay[i+1]];
		}
	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{
				weights[i][j][k] = rn1.generate();
				if(weights[i][j][k]<(rn1.generate())*rn1.generate())
					weights[i][j][k]=-1*weights[i][j][k];
				xerox_weights[i][j][k]=weights[i][j][k];
			}

	deli = new double * [no_of_layers];
	for(int i=0;i<no_of_layers;i++)
		deli[i] = new double [nonperlay[i]];
	itercnt=0;
	annealing_threshold=-1;
	error= new double[iter];
}

void VVNetwork :: VVreadinputfrom(char * fn)
{
	fstream g(fn,ios::in);

	int i=0;
	double t;
	while(g)
	{
		g>>t;
		if(g)
			i++;
	}
	g.close();
	
	fstream f(fn,ios::in);
	inputs=new double*[i/nonperlay[0]];
	for(int i1=0;i1<i/nonperlay[0];i1++)
		inputs[i1]=new double[nonperlay[0]];
	no_training_records=i/nonperlay[0];
	i=0;
	//so the array is inputs[no_records][no_bits]
	int j=0;
	while(f)
	{
		f>>t;
		if(f)
		{
			inputs[j][i]=t;
			if(((i+1)%nonperlay[0])==0){i=0;j++;}
			else
				i++;	
		}

	}
	f.close();
}

void VVNetwork :: VVreadoutputfrom(char*fn)
{

	fstream g(fn,ios::in);
	int i=0;
	double t;
	while(g)
	{
		g>>t;
		if(g)
			i++;
	}
	no_output_records=i/nonperlay[no_of_layers-1];
	g.close();
	
	fstream f(fn,ios::in);
	i=0;

	outputs = new double [no_output_records*nonperlay[no_of_layers-1]];

	while(f)
	{

		f >> t;
		if(f)
		{
			outputs[i]=t;
			i++;
		}

	}
	f.close();


}
void VVNetwork :: VVFeedForward(int &i)
{
	//feed forward part

	for(int j=0;j<nonperlay[0];j++)//loop feeds the inputs to layer 0
	{
		layers[0].vvset_perceptron(j,inputs[i][j]);

	}
	for(int k=1;k<no_of_layers;k++)//feeding the middle men
	{
		for(int l=0;l<nonperlay[k];l++)
		{
			double S=0;
			for(int m=0;m<nonperlay[k-1];m++)
			{S+=(weights[k-1][m][l]*layers[k-1].vvget_perceptron(m));}
			if(layers[k].vvbiased(l))
			{
				S+=layers[k].vvbiased(l);
				S*=1.0/(1+nonperlay[k-1]);
			}
			else S*=1.0/(nonperlay[k-1]);
			layers[k].vvset_perceptron(l,vvtransfer_function(S));
			layers[k].vvset_perceptron_accumulator(l,S);


		}

	}
}
void VVNetwork :: VVBackPropagate(int &z)
{
	for(int j=no_of_layers-1;j>0;j--)	
	{
		for(int k=0;k<nonperlay[j];k++)
		{


			if(z>=no_output_records*nonperlay[no_of_layers-1])z=0;

			if(j==no_of_layers-1)	//check if output layer...then
			{
				deli[j][k] = (outputs[z++] - layers[no_of_layers-1].vvget_perceptron(k))*(vvtransfer_function_derivative(layers[no_of_layers-1].vvget_perceptron_accumulator(k)));
				global_error += deli[j][k];
				// error = (T - f(x)) * (deriv(x)) 			
			}


			else
			{

				double S=0;
				for(int l=0;l<nonperlay[j+1];l++)
					S+=deli[j+1][l] * weights[j][k][l];

				deli[j][k] = S * (vvtransfer_function_derivative(layers[j].vvget_perceptron_accumulator(k)));

			}	

			for(int l=0;l<nonperlay[j-1];l++)
			{
				double delw = (learning_rate * deli[j][k] * layers[j-1].vvget_perceptron(l));
				xerox_weights[j-1][l][k] +=delw;
			}

			if(layers[j].vvbiased(k))
			{
				float delw = (deli[j][k]*learning_rate);
				layers[j].vvlayer_bias_perceprton(k,layers[j].vvbiased(k)+delw);
			}

		}
	}
	for(int ii=0;ii<no_of_layers-1;ii++)
		for(int jj=0;jj<nonperlay[ii];jj++)
			for(int kk=0;kk<nonperlay[ii+1];kk++)
			{
				weights[ii][jj][kk]=xerox_weights[ii][jj][kk];

			}


}

void VVNetwork :: VVnetwork_train()
{
	double temp=10;
	int iteratorsample=0;
//	for(int iteratorsample=0;iteratorsample<iter;iteratorsample++)
	{
		if(progbar)
			VVProgressBar(iteratorsample);
		int z=0;
		for(int i=0;i<no_training_records;i++) //loop to traverse entire input array
		{
			VVFeedForward(i);
			VVBackPropagate(z);
	}
	if(annealing_threshold>0)
	{
		if(global_error/nonperlay[no_of_layers-1]-temp<annealing_threshold)
		{
			temp=global_error/nonperlay[no_of_layers-1];
			learning_rate-=1;
		}
		else learning_rate+=1;
	}
	error[itercnt]=global_error;
	global_error=0;
	itercnt++;
	//	show_weights();
}

}
vector <double> VVNetwork :: VVtest_network(char fn[])
{

	vector <double> outputvals;

	fstream g(fn,ios::in|ios::binary);
	double t;
	int i2=0,i3=0;
	while(g)
	{
		g>>t;
		if(g)i2++;
	}
	g.close();
	

	fstream f(fn,ios::in|ios::binary);
	for(int count=0;count<i2;count+=nonperlay[0])
	{
		for(i3=0;i3<nonperlay[0];i3++)
		{

			f>>t;
			if(f)
			{
				layers[0].vvset_perceptron(i3,t);
			}
		}
		for(int k=1;k<no_of_layers;k++)//feeding the middle men
		{
			for(int l=0;l<nonperlay[k];l++)
			{

				double S=0;
				for(int m=0;m<nonperlay[k-1];m++)
				{S+=weights[k-1][m][l]*layers[k-1].vvget_perceptron(m);}


				if(layers[k].vvbiased(l))
				{
					S+=layers[k].vvbiased(l);
					S*=1.0/(1+nonperlay[k-1]);
				}
				else S*=1.0/(nonperlay[k-1]);

				layers[k].vvset_perceptron(l,vvtransfer_function(S));

				layers[k].vvset_perceptron_accumulator(l,S);
			}

		}



		for(int i=0;i<nonperlay[no_of_layers-1];i++)
			outputvals.push_back(layers[no_of_layers-1].vvget_perceptron(i));//<<' ';

	}



	//	cout << "MANUAL CALCs : " << endl;
	//	cout << vvtransfer_function(((weights[0][0][0] * 0) + (weights[0][1][0] * 1)+(layers[1].vvbiased(0)))/1.0) << endl;
	//show_weights();

	return (outputvals);

}



int VVNetwork :: VVsave_network(char fn[])
{

	fstream f(fn,ios::out|ios::binary);
	
	f.write((char *)&no_of_layers,sizeof(int));

	
	for(int i=0;i<no_of_layers;i++)
		{
			f.write((char *)&nonperlay[i],sizeof(nonperlay[i]));
					}


	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{
				f.write((char *)&weights[i][j][k],sizeof(weights[i][j][k]));
			
			}

	for(int i=1;i<no_of_layers;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			double temp = layers[i].vvbiased(j);
			f.write((char *)&temp,sizeof(double));
		
		}


	double temp = layers[1].vvbiased(0);
	f.write((char *)&temp,sizeof(double));

	temp = layers[1].vvbiased(1);
	f.write((char *)&temp,sizeof(double));

	temp = layers[2].vvbiased(0);
	f.write((char *)&temp,sizeof(double));




	f.write((char *)&learning_rate,sizeof(learning_rate));
	f.write((char *)&momentum,sizeof(momentum));
	

	f.close();

	//show_weights();

}

int VVNetwork :: VVload_network(char fn[])
{

	
	no_of_layers = 0;
	fstream f(fn,ios::in|ios::binary);
	f.read((char *)&no_of_layers,sizeof(no_of_layers));
	vector<int> npl;
	for(int i=0;i<no_of_layers;i++)
	{   
		int temp;
		f.read((char *)&temp,sizeof(temp));
		npl.push_back(temp);
	}
	
	error = NULL;
	inputs = NULL;
	outputs = NULL;
	nonperlay = npl;

	layers = new VVLayer[no_of_layers];

	for(int i=0;i<no_of_layers;i++)
		layers[i].vvlayerinit(nonperlay[i]);

	deli = new double * [no_of_layers];

	for(int i=0;i<no_of_layers;i++)
		deli[i] = new double [nonperlay[i]];


	weights=new double**[no_of_layers-1];
	xerox_weights=new double**[no_of_layers-1];


	for(int i=0;i<no_of_layers-1;i++)
	{
		weights[i]=new double*[nonperlay[i]];
		xerox_weights[i] = new double*[nonperlay[i]];
	}

	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			weights[i][j]= new double [nonperlay[i+1]];
			xerox_weights[i][j]= new double[nonperlay[i+1]];
		}

	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{

				f.read((char *)&weights[i][j][k],sizeof(weights[i][j][k]));

			}

	for(int i=1;i<no_of_layers;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			double temp;
			f.read((char *)&temp,sizeof(double));
			layers[i].vvlayer_bias_perceprton(j,temp);
		}

	f.read((char *)&learning_rate,sizeof(learning_rate));
	f.read((char *)&momentum,sizeof(momentum));


	f.close();


}



//------------------------------------------------Time series Network------------------------

void TSVVNetwork :: show_weights()
{

	cout << "Weights: " << endl;
	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{
				cout << weights[i][j][k] << endl;

			}
}

void TSVVNetwork :: vvautobias()
{
	randoom r1(1,12344*time(0));
	r1.generate();
	for(int i=1;i<no_of_layers;i++)
		for(int j=0;j<nonperlay[i];j++)
		{

			double val = r1.generate();
			if(val*r1.generate()<r1.generate()) val*=-1.0;
			layers[i].vvlayer_bias_perceprton(j,val);
		}
}
void TSVVNetwork :: VVProgressBar(int i=1)
{
	if(progbar)   
	{
		if(i%(iter/100)==0)
		{
			if(i/(iter/100)==99)
				cout<<'|'<<endl;				
			else
			cout<<'*'<<flush;
		}	
	}		
	else 
	{
		cout<<endl<<'|';
		progbar=1;
	}
}
void TSVVNetwork :: VVSimulatedAnnealing(double thres)
{
	annealing_threshold=thres;
}
TSVVNetwork :: TSVVNetwork(int a,...)
{

	va_list ap;
	va_start(ap, a);
	layers = new VVLayer[a];
	for(int i=0;i<a;i++)
		nonperlay.push_back(va_arg(ap, int));	
	for(int i=0;i<a;i++)
		layers[i].vvlayerinit(nonperlay[i]);
	no_of_layers=a;
	learning_rate=va_arg(ap,double);
	momentum=va_arg(ap,double);
	iter=va_arg(ap,int);
	randoom rn1(1,67435*time(0)+a);
	rn1.generate();
	progbar=0;
	weights=new double**[no_of_layers-1];
	xerox_weights=new double**[no_of_layers-1];

	for(int i=0;i<no_of_layers-1;i++)
	{
		weights[i]=new double*[nonperlay[i]];
		xerox_weights[i] = new double*[nonperlay[i]];
	}
	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			weights[i][j]= new double [nonperlay[i+1]];
			xerox_weights[i][j]= new double[nonperlay[i+1]];
		}
	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{
				weights[i][j][k] = rn1.generate();
				if(weights[i][j][k]<(rn1.generate())*rn1.generate())
					weights[i][j][k]=-1*weights[i][j][k];
				xerox_weights[i][j][k]=weights[i][j][k];
			}

	deli = new double * [no_of_layers];
	for(int i=0;i<no_of_layers;i++)
		deli[i] = new double [nonperlay[i]];
	itercnt=0;
	annealing_threshold=-1;
	error= new double[iter];
}

void TSVVNetwork :: VVreadinputfrom(char * fn)
{
	fstream f(fn,ios::in);

	int i=1;
	double t;
	if(progbar)
		cout<<"The input data from file : "<<fn<<endl;
	double max;
	f>>max;
	while(f)
	{
		f>>t;
		if(f)
		{
			i++;
			if(t>max) max =t;
		}
	}
	f.close();
	f.open(fn,ios::in);
	inputs=new double*[i];
	for(int i1=0;i1<i;i1++)
		inputs[i1]=new double;
	no_training_records=i;//nonperlay[0];
	i=0;
	//so the array is inputs[no_records][no_bits]
			int j=0;
	while(f)
	{
		f>>t;
		if(f)
		{
			inputs[i][0]=t/max;
			//if(((i+1)%nonperlay[0])==0){i=0;j++;}
			//else
				i++;	
		}

	}
	f.close();
}

void TSVVNetwork :: VVreadoutputfrom(char*fn)
{

	
	no_output_records = no_training_records - nonperlay[0];
	outputs = new double [no_output_records*nonperlay[no_of_layers-1]];
	for(int i=0;i<no_output_records;i++)
		outputs[i]=inputs[i+nonperlay[0]][0];
		

	
	


}

void TSVVNetwork :: VVFeedForward(int &i)
{
	//feed fàorward part

	for(int j=0;j<nonperlay[0];j++)//loop feeds the inputs to layer 0
	{
		layers[0].vvset_perceptron(j,inputs[j+i][0]);
	//	cout << "Inputs: " << inputs[j+i][0] << '\t';
		
	}

	for(int k=1;k<no_of_layers;k++)//feeding the middle men
	{
		for(int l=0;l<nonperlay[k];l++)
		{
			double S=0;
			for(int m=0;m<nonperlay[k-1];m++)
			{S+=(weights[k-1][m][l]*layers[k-1].vvget_perceptron(m));}

			if(layers[k].vvbiased(l))
			{
				S+=layers[k].vvbiased(l);
				S*=1.0/(1+nonperlay[k-1]);
			}
			else S*=1.0/(nonperlay[k-1]);
			layers[k].vvset_perceptron(l,vvtransfer_function(S));
			layers[k].vvset_perceptron_accumulator(l,S);
		//	if(k==no_of_layers-1)cout << '\t' << "Output: " << layers[k].vvget_perceptron(l) << endl;

		}
			//show_weights();
		//	cout << layers[no_of_layers-1].vvbiased(0) << endl;
	}
}
void TSVVNetwork :: VVBackPropagate(int &z)
{
	
	


	for(int j=no_of_layers-1;j>0;j--)	
	{
		for(int k=0;k<nonperlay[j];k++)
		{


			if(z>=no_output_records*nonperlay[no_of_layers-1])z=0;

			if(j==no_of_layers-1)	//check if output layer...then
			{
				
							deli[j][k] = (outputs[z++] - layers[no_of_layers-1].vvget_perceptron(k))*(vvtransfer_function_derivative(layers[no_of_layers-1].vvget_perceptron_accumulator(k)));
			//	cout << layers[no_of_layers-1].vvget_perceptron_accumulator(k) << '\t' << vvtransfer_function_derivative(layers[no_of_layers-1].vvget_perceptron_accumulator(k),48.0) << endl;
				global_error += deli[j][k];
		//		cout << "Output: " <<'\t' << outputs[z-1] << endl; 
				// error = (T - f(x)) * (deriv(x)) 			
			}


			else
			{

				double S=0;
				for(int l=0;l<nonperlay[j+1];l++)
					S+=deli[j+1][l] * weights[j][k][l];

				deli[j][k] = S * (vvtransfer_function_derivative(layers[j].vvget_perceptron_accumulator(k)));

			}	

			for(int l=0;l<nonperlay[j-1];l++)
			{
				double delw = (learning_rate * deli[j][k] * layers[j-1].vvget_perceptron(l));
				//cout << deli[j][k] << '\t' ;
				xerox_weights[j-1][l][k] +=delw;
			}
			if(layers[j].vvbiased(k))
			{
				float delw = (deli[j][k]*learning_rate);
				layers[j].vvlayer_bias_perceprton(k,layers[j].vvbiased(k)+delw);
			}

		}
	}
	for(int ii=0;ii<no_of_layers-1;ii++)
		for(int jj=0;jj<nonperlay[ii];jj++)
			for(int kk=0;kk<nonperlay[ii+1];kk++)
			{
				weights[ii][jj][kk]=xerox_weights[ii][jj][kk];

			}


}

void TSVVNetwork :: VVnetwork_train()
{
	double temp=10;
	for(int iteratorsample=0;iteratorsample<iter;iteratorsample++)
	{
		//if(progbar)
			//VVProgressBar(iteratorsample);
		int z=0;
		for(int i=0;i<=no_training_records-nonperlay[0]-1;i++) //loop to traverse entire input array
		{
			VVFeedForward(i);
			VVBackPropagate(z);
		}
	if(annealing_threshold>0)
	{
		if(global_error/nonperlay[no_of_layers-1]-temp<annealing_threshold)
		{
			temp=global_error/nonperlay[no_of_layers-1];
			learning_rate-=1;
		}
		else learning_rate+=1;
	}
	error[itercnt]=global_error;
	global_error=0;
	itercnt++;
	//	show_weights();
}
cout<<endl<<"Training complete\n";

}
vector <double> TSVVNetwork :: VVtest_network(char fn[])
{

	vector <double> outputvals;

	fstream f(fn,ios::in|ios::binary);
	double t;
	int i2=0,i3=0;
	while(f)
	{
		f>>t;
		if(f)i2++;
	}
	f.close();
	f.open(fn,ios::in|ios::binary);
	for(int count=0;count<i2;count+=nonperlay[0])
	{
		for(i3=0;i3<nonperlay[0];i3++)
		{

			f>>t;
			if(f)
			{
				layers[0].vvset_perceptron(i3,t/48.0);
			}
		}
		for(int k=1;k<no_of_layers;k++)//feeding the middle men
		{
			for(int l=0;l<nonperlay[k];l++)
			{

				double S=0;
				for(int m=0;m<nonperlay[k-1];m++)
				{S+=weights[k-1][m][l]*layers[k-1].vvget_perceptron(m);}


				if(layers[k].vvbiased(l))
				{
					S+=layers[k].vvbiased(l);
					S*=1.0/(1+nonperlay[k-1]);
				}
				else S*=1.0/(nonperlay[k-1]);

				layers[k].vvset_perceptron(l,vvtransfer_function(S));

				layers[k].vvset_perceptron_accumulator(l,S);
			}

		}

		for(int i=0;i<nonperlay[no_of_layers-1];i++)
			outputvals.push_back(layers[no_of_layers-1].vvget_perceptron(i));//<<' ';

	}


	//	cout << "MANUAL CALCs : " << endl;
	//	cout << vvtransfer_function(((weights[0][0][0] * 0) + (weights[0][1][0] * 1)+(layers[1].vvbiased(0)))/1.0) << endl;
	//show_weights();

	return (outputvals);

}

void TSVVNetwork :: VVsave_network(char fn[])
{

	fstream f(fn,ios::out|ios::binary);
	f.write((char *)&no_of_layers,sizeof(no_of_layers));

	for(int i=0;i<no_of_layers;i++)
		f.write((char *)&nonperlay[i],sizeof(nonperlay[i]));

	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{
				f.write((char *)&weights[i][j][k],sizeof(weights[i][j][k]));
			}

	for(int i=1;i<no_of_layers;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			double temp = layers[i].vvbiased(j);
			f.write((char *)&temp,sizeof(double));
		}

	f.write((char *)&learning_rate,sizeof(learning_rate));
	f.write((char *)&momentum,sizeof(momentum));



	f.close();

	//show_weights();

}

int TSVVNetwork :: VVload_network(char fn[])
{

	vector<int> npl;

	fstream f(fn,ios::in|ios::binary);
	f.read((char *)&no_of_layers,sizeof(no_of_layers));

	for(int i=0;i<no_of_layers;i++)
	{   
		int temp;
		f.read((char *)&temp,sizeof(temp));
		npl.push_back(temp);
	}


	inputs = NULL;
	outputs = NULL;
	nonperlay = npl;

	layers = new VVLayer[no_of_layers];

	for(int i=0;i<no_of_layers;i++)
		layers[i].vvlayerinit(nonperlay[i]);

	deli = new double * [no_of_layers];

	for(int i=0;i<no_of_layers;i++)
		deli[i] = new double [nonperlay[i]];


	weights=new double**[no_of_layers-1];
	xerox_weights=new double**[no_of_layers-1];


	for(int i=0;i<no_of_layers-1;i++)
	{
		weights[i]=new double*[nonperlay[i]];
		xerox_weights[i] = new double*[nonperlay[i]];
	}

	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			weights[i][j]= new double [nonperlay[i+1]];
			xerox_weights[i][j]= new double[nonperlay[i+1]];
		}

	for(int i=0;i<no_of_layers-1;i++)
		for(int j=0;j<nonperlay[i];j++)
			for(int k=0;k<nonperlay[i+1];k++)
			{

				f.read((char *)&weights[i][j][k],sizeof(weights[i][j][k]));

			}

	for(int i=1;i<no_of_layers;i++)
		for(int j=0;j<nonperlay[i];j++)
		{
			double temp;
			f.read((char *)&temp,sizeof(double));
			layers[i].vvlayer_bias_perceprton(j,temp);
		}

	f.read((char *)&learning_rate,sizeof(learning_rate));
	f.read((char *)&momentum,sizeof(momentum));


	f.close();


}
