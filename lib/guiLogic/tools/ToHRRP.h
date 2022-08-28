//
// MATLAB Compiler: 8.4 (R2022a)
// Date: Thu Aug 11 11:06:41 2022
// Arguments: "-B""macro_default""-W""cpplib:ToHRRP""-T""link:lib""ToHRRP.m""-C"
//

#ifndef ToHRRP_h
#define ToHRRP_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_ToHRRP_C_API 
#define LIB_ToHRRP_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_ToHRRP_C_API 
bool MW_CALL_CONV ToHRRPInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_ToHRRP_C_API 
bool MW_CALL_CONV ToHRRPInitialize(void);

extern LIB_ToHRRP_C_API 
void MW_CALL_CONV ToHRRPTerminate(void);

extern LIB_ToHRRP_C_API 
void MW_CALL_CONV ToHRRPPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_ToHRRP_C_API 
bool MW_CALL_CONV mlxToHRRP(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_ToHRRP
#define PUBLIC_ToHRRP_CPP_API __declspec(dllexport)
#else
#define PUBLIC_ToHRRP_CPP_API __declspec(dllimport)
#endif

#define LIB_ToHRRP_CPP_API PUBLIC_ToHRRP_CPP_API

#else

#if !defined(LIB_ToHRRP_CPP_API)
#if defined(LIB_ToHRRP_C_API)
#define LIB_ToHRRP_CPP_API LIB_ToHRRP_C_API
#else
#define LIB_ToHRRP_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_ToHRRP_CPP_API void MW_CALL_CONV ToHRRP(int nargout, mwArray& retn, const mwArray& sPath, const mwArray& dPath);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
