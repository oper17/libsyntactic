#include <syntactic/syntactic.h>
#include <iostream>
#include <vector>
using namespace std;

void train()
{
		int times=3000;
	VVNetwork N1(3,2,2,1,0.91,0.2,times);
	N1.vvtransfer_function = VV_SIGMOID_HALF;
	N1.vvtransfer_function_derivative = VV_SIGMOID_HALF_DERIV;
	N1.VVreadinputfrom("xor.in");
	N1.VVreadoutputfrom("xor.out");
	N1.vvautobias();
	
 
	for(int i=0;i<times;i++)   
	{ 
		N1.VVnetwork_train();
		if(i%(times/100)==0) cout <<"% "<< i/(times/100)<<" over."<<endl;
	}
	N1.VVsave_network("xor.nnw");
}

void test()
{
	VVNetwork N2;
	N2.VVload_network("xor.nnw");
	N2.vvtransfer_function = VV_SIGMOID_HALF;
	N2.vvtransfer_function_derivative = VV_SIGMOID_HALF_DERIV;
	vector<double> out;
	

	out=N2.VVtest_network("test");
	for(int i=0;i<out.size();i++)
	{
		
		cout<<out[i]<<" ";
	}
	
	cout<<endl;

}

int main(int argc,char ** argv)
{
	
	
	if(argc>1) 
	{
		string temp = argv[1];
		if(temp=="train") train();
		else if(temp=="test")test();
		else cout << "Syntax: xor <train/test> " << endl;
	}
	else cout << "Syntax: xor <train/test> " << endl;
	

}
	
	
