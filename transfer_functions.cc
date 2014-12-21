#include "transfer_functions.h"

double vvtransfer_function_hardlimiter(double x)
{
	if (x<0) return -1;
	else return 1;
}

double vvtransfer_function_hardlimiter_derivative(double x)
{
	return 0;
}

double vvtransfer_function_ramp(double x)
{
	if (x<0) return 0;
	else if(x<=1) return x;
	else return 1;
}
double vvtransfer_function_ramp_derivative(double x)
{
	if((x>=0) && (x<=1)) return 1;
	else return 0;
}
double vvtransfer_function_sigmoid_half(double x)
{
	return 1.0/(1+exp(-1.0*x));
}

double vvtransfer_function_sigmoid_half_derivative(double x)
{
	return ((1.0 - vvtransfer_function_sigmoid_half(x))*(vvtransfer_function_sigmoid_half(x)));
}

double vvtransfer_function_sigmoid_full(double x)
{
	if(x>=0)
	return (1.0 - 1.0/(1.0+x));
	else return (-1.0 + 1.0/(1.0 -x));
}
double vvtransfer_function_sigmoid_full_derivative(double x)
{	
	if(x>=0)return (1.0/((1.0+x)*(1.0+x)));
	else return (1.0/((1.0-x)*(1.0-x)));
}
double vvtransfer_function_tanh(double x)
{
	return(tanh(x));
}	
double vvtransfer_function_tanh_derivative(double x)
{
	return(	1-(tanh(x)*tanh(x)));
}
