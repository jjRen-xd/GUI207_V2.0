#ifndef TRTINFER_H
#define TRTINFER_H

#include <QObject>
#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <QThread>
#include "libtorchTest.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include <mat.h>

class TrtInfer
{
public:
    TrtInfer(std::map<std::string, int> class2label);
public slots:

    void testOneSample(std::string targetPath, int emIndex, std::string modelPath, int &predIdx,std::vector<float> &degrees);
private:
    nvinfer1::IBuilder* builder{ nullptr };
    nvinfer1::INetworkDefinition* network{ nullptr };
    nvinfer1::IBuilderConfig* config{ nullptr };
    nvinfer1::IHostMemory* modelStream{ nullptr };
    nvinfer1::ICudaEngine* engine{ nullptr };
    nvinfer1::IExecutionContext* context{ nullptr };
    bool isDynamic{true};//模型是否是动态batch
    int batchSize{1};
    std::vector<int> inputdims; std::vector<int> outputdims;
    int input_len{1}; int output_len{1};//不包含batchsize
    // 不同平台下文件夹搜索工具
    std::map<std::string, int> class2label;

    void doInference(nvinfer1::IExecutionContext&context,  float* input, float* output);

};

#endif // TRTINFER_H
