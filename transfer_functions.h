/** \file transfer_functions.h
 * \brief This is the header file containing transfer function information.
 * 
 * This file defines macros and functions they connect to.
 * These macros and functions are used to specify the neural network's transfer fucntions
 * and transfer function derivatives using vvtransfer_function and vvtransfer_function_derivative.
 */

#include <math.h>

#define VV_HARDLIMIT vvtransfer_function_hardlimiter
#define VV_HARDLIMIT_DERIV vvtransfer_function_hardlimiter_derivative
#define VV_SIGMOID_HALF vvtransfer_function_sigmoid_half
#define VV_SIGMOID_HALF_DERIV vvtransfer_function_sigmoid_half_derivative
#define VV_SIGMOID_FULL vvtransfer_function_sigmoid_full
#define VV_SIGMOID_FULL_DERIV vvtransfer_function_sigmoid_full_derivative
#define VV_RAMP vvtransfer_function_ramp
#define VV_RAMP_DERIV vvtransfer_function_ramp_derivative
#define VV_HYPERBOLIC vvtransfer_function_tanh
#define VV_HYPERBOLIC_DERIV vvtransfer_function_tanh_derivative

/** \brief Hardlimiter Sigmoid transfer function.
 *
 * This transfer function is hardlimited above 1.
 * Its corresponding macro is: VV_HARDLIMIT.
 */
double vvtransfer_function_hardlimiter(double x);
/** \brief Hardlimiter Sigmoid transfer function derivative.
 *
 * The corresponding macro is: VV_HARDLIMIT_DERIV.
 */
double vvtransfer_function_hardlimiter_derivative(double x);
/** \brief Ramp Sigmoid transfer function.
 *
 * This transfer function is a ramp signal.
 * Its corresponding macro is: VV_RAMP.
 */
double vvtransfer_function_ramp(double x);
/** \brief Ramp Sigmoid transfer function derivative.
 *
 * The corresponding macro is: VV_RAMP_DERIV.
 */
double vvtransfer_function_ramp_derivative(double x);
/** \brief  Sigmoid half transfer function.
 *
 * This transfer function is an S signal from 0 to 1.
 * Its corresponding macro is: VV_SIGMOID_HALF.
 */
double vvtransfer_function_sigmoid_half(double x);
/** \brief Sigmoid half transfer function derivative.
 *
 * The corresponding macro is: VV_SIGMOID_HALF_DERIV.
 */
double vvtransfer_function_sigmoid_half_derivative(double x);
/** \brief Sigmoid full transfer function.
 *
 * This transfer function is an S signal from -1 to 1.
 * Its corresponding macro is: VV_SIGMOID_FULL.
 */
double vvtransfer_function_sigmoid_full(double x);
/** \brief Sigmoid full transfer function derivative.
 *
 * The corresponding macro is: VV_SIGMOID_FULL_DERIV.
 */
double vvtransfer_function_sigmoid_full_derivative(double x);
/** \brief Hyperbolic Sigmoid transfer function.
 *
 * This transfer function is a tanh curve.
 * Its corresponding macro is: VV_HYPERBOLIC.
 */
double vvtransfer_function_tanh(double x);
/** \brief Hyperbolic Sigmoid transfer function derivative.
 *
 * The corresponding macro is: VV_HYPERBOLIC_DERIV.
 */
double vvtransfer_function_tanh_derivative(double x);
