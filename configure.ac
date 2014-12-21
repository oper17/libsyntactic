#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
ECHO "INSTALLING SYNTACTIC NEURAL LIBRARY (OFFSHOOT OF VeVa)"
AC_PREREQ(2.59)
AC_INIT(Syntactic, 0.1.2, BUG-REPORT-ADDRESS)
AC_PROG_RANLIB
AM_PROG_LIBTOOL 
AM_INIT_AUTOMAKE

AC_CONFIG_SRCDIR([])
AC_CONFIG_HEADER([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.
AC_CHECK_LIB([gsl],[gsl_blas_ccopy])
AC_CHECK_LIB([gslcblas],[cblas_ccopy])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST

# Checks for library functions.
AC_CHECK_FUNCS([atexit sqrt])

#AC_CONFIG_FILES([Makefile
#		src/splash/Makefile])
AC_OUTPUT(Makefile)
