//
// MATLAB Compiler: 8.4 (R2022a)
// Date: Thu Aug 11 11:06:41 2022
// Arguments: "-B""macro_default""-W""cpplib:ToHRRP""-T""link:lib""ToHRRP.m""-C"
//

#define EXPORTING_ToHRRP 1
#include "ToHRRP.h"

static HMCRINSTANCE _mcr_inst = NULL; /* don't use nullptr; this may be either C or C++ */

#if defined( _MSC_VER) || defined(__LCC__) || defined(__MINGW64__)
#ifdef __LCC__
#undef EXTERN_C
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
#undef interface

static char path_to_dll[_MAX_PATH];

BOOL WINAPI DllMain(HINSTANCE hInstance, DWORD dwReason, void *pv)
{
    if (dwReason == DLL_PROCESS_ATTACH)
    {
        if (GetModuleFileName(hInstance, path_to_dll, _MAX_PATH) == 0)
            return FALSE;
    }
    else if (dwReason == DLL_PROCESS_DETACH)
    {
    }
    return TRUE;
}
#endif
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

static int mclDefaultPrintHandler(const char *s)
{
    return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern C block */
#endif

#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

static int mclDefaultErrorHandler(const char *s)
{
    int written = 0;
    size_t len = 0;
    len = strlen(s);
    written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
    if (len > 0 && s[ len-1 ] != '\n')
        written += mclWrite(2 /* stderr */, "\n", sizeof(char));
    return written;
}

#ifdef __cplusplus
} /* End extern C block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_ToHRRP_C_API
#define LIB_ToHRRP_C_API /* No special import/export declaration */
#endif

LIB_ToHRRP_C_API 
bool MW_CALL_CONV ToHRRPInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
    if (_mcr_inst)
        return true;
    if (!mclmcrInitialize())
        return false;
    if (!GetModuleFileName(GetModuleHandle("ToHRRP"), path_to_dll, _MAX_PATH))
        return false;
    bResult = mclInitializeComponentInstanceNonEmbeddedStandalone(&_mcr_inst,
        path_to_dll,
        "ToHRRP",
        LibTarget,
        error_handler, 
        print_handler);
    if (!bResult)
    return false;
    return true;
}

LIB_ToHRRP_C_API 
bool MW_CALL_CONV ToHRRPInitialize(void)
{
    return ToHRRPInitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_ToHRRP_C_API 
void MW_CALL_CONV ToHRRPTerminate(void)
{
    if (_mcr_inst)
        mclTerminateInstance(&_mcr_inst);
}

LIB_ToHRRP_C_API 
void MW_CALL_CONV ToHRRPPrintStackTrace(void) 
{
    char** stackTrace;
    int stackDepth = mclGetStackTrace(&stackTrace);
    int i;
    for(i=0; i<stackDepth; i++)
    {
        mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
        mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
    }
    mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_ToHRRP_C_API 
bool MW_CALL_CONV mlxToHRRP(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    return mclFeval(_mcr_inst, "ToHRRP", nlhs, plhs, nrhs, prhs);
}

LIB_ToHRRP_CPP_API 
void MW_CALL_CONV ToHRRP(int nargout, mwArray& retn, const mwArray& sPath, const mwArray& 
                         dPath)
{
    mclcppMlfFeval(_mcr_inst, "ToHRRP", nargout, 1, 2, &retn, &sPath, &dPath);
}

