//
// MATLAB Compiler: 8.4 (R2022a)
// Date: Tue Oct 18 14:00:18 2022
// Arguments: "-B""macro_default""-W""cpplib:ToHrrp""-T""link:lib""ToHrrp.m""-C"
//

#define EXPORTING_ToHrrp 1
#include "ToHrrp.h"

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
#ifndef LIB_ToHrrp_C_API
#define LIB_ToHrrp_C_API /* No special import/export declaration */
#endif

LIB_ToHrrp_C_API 
bool MW_CALL_CONV ToHrrpInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
    if (_mcr_inst)
        return true;
    if (!mclmcrInitialize())
        return false;
    if (!GetModuleFileName(GetModuleHandle("ToHrrp"), path_to_dll, _MAX_PATH))
        return false;
    bResult = mclInitializeComponentInstanceNonEmbeddedStandalone(&_mcr_inst,
        path_to_dll,
        "ToHrrp",
        LibTarget,
        error_handler, 
        print_handler);
    if (!bResult)
    return false;
    return true;
}

LIB_ToHrrp_C_API 
bool MW_CALL_CONV ToHrrpInitialize(void)
{
    return ToHrrpInitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_ToHrrp_C_API 
void MW_CALL_CONV ToHrrpTerminate(void)
{
    if (_mcr_inst)
        mclTerminateInstance(&_mcr_inst);
}

LIB_ToHrrp_C_API 
void MW_CALL_CONV ToHrrpPrintStackTrace(void) 
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


LIB_ToHrrp_C_API 
bool MW_CALL_CONV mlxToHrrp(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    return mclFeval(_mcr_inst, "ToHrrp", nlhs, plhs, nrhs, prhs);
}

LIB_ToHrrp_CPP_API 
void MW_CALL_CONV ToHrrp(int nargout, mwArray& retn, const mwArray& sPath, const mwArray& 
                         dPath, const mwArray& sName)
{
    mclcppMlfFeval(_mcr_inst, "ToHrrp", nargout, 1, 3, &retn, &sPath, &dPath, &sName);
}

