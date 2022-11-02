//
// MATLAB Compiler: 8.4 (R2022a)
// Date: Tue Oct 18 14:00:18 2022
// Arguments: "-B""macro_default""-W""cpplib:ToHrrp""-T""link:lib""ToHrrp.m""-C"
//

#ifndef ToHrrp_h
#define ToHrrp_h 1

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
#ifndef LIB_ToHrrp_C_API 
#define LIB_ToHrrp_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_ToHrrp_C_API 
bool MW_CALL_CONV ToHrrpInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_ToHrrp_C_API 
bool MW_CALL_CONV ToHrrpInitialize(void);

extern LIB_ToHrrp_C_API 
void MW_CALL_CONV ToHrrpTerminate(void);

extern LIB_ToHrrp_C_API 
void MW_CALL_CONV ToHrrpPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_ToHrrp_C_API 
bool MW_CALL_CONV mlxToHrrp(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_ToHrrp
#define PUBLIC_ToHrrp_CPP_API __declspec(dllexport)
#else
#define PUBLIC_ToHrrp_CPP_API __declspec(dllimport)
#endif

#define LIB_ToHrrp_CPP_API PUBLIC_ToHrrp_CPP_API

#else

#if !defined(LIB_ToHrrp_CPP_API)
#if defined(LIB_ToHrrp_C_API)
#define LIB_ToHrrp_CPP_API LIB_ToHrrp_C_API
#else
#define LIB_ToHrrp_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_ToHrrp_CPP_API void MW_CALL_CONV ToHrrp(int nargout, mwArray& retn, const mwArray& sPath, const mwArray& dPath, const mwArray& sName);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
