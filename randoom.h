/** \file randoom.h
 * \brief This is the header file containing all the classes and user variables.
 * 
 * This file contains the following classes:
 * class randoom 
 */

#include <gsl/gsl_rng.h>
#include<time.h>

/** \brief This class for random number generation.
 *
 * This class uses gsl for pseudo random number generation.
 * To learn more, visit the gsl homepage. The function generate(),
 * generates the required random number.
 */
class randoom
{
  const gsl_rng_type * T;
  gsl_rng * r;
	public:
	
	randoom(int type,unsigned long seed)
	{
	  gsl_rng_env_setup();
	  switch (type)
	  {
	  	case 0 :  T = gsl_rng_taus; break;
		case 1 : T = gsl_rng_mt19937;break;
		case 2 : T = gsl_rng_ranlux389;break;
		case 3 : T = gsl_rng_cmrg;break;
		default : T =gsl_rng_taus;break;
	  }
	  r = gsl_rng_alloc (T);
	  gsl_rng_set(r,seed*time(0));
	}

	double generate()
	{	
	
		double u = gsl_rng_uniform (r);
		return (u);
	}

	~randoom()
	{
	  gsl_rng_free (r);
	}
 
};
