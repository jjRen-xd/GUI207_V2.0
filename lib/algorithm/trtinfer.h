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
//#include "libtorchTest.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include <mat.h>

class TrtInfer
{
public:
    TrtInfer(std::map<std::string, int> class2label);
    void setBatchSize(int batchSize);//留出来的接口
public slots:

    void testOneSample(std::string targetPath, int emIndex, std::string modelPath, int *predIdx,std::vector<float> &degrees);
    bool testAllSample(std::string dataset_path,std::string model_path,int inferBatch, float &Acc,std::vector<std::vector<int>> &confusion_matrix);

private:
    nvinfer1::IBuilder* builder{ nullptr };
    nvinfer1::INetworkDefinition* network{ nullptr };
    nvinfer1::IBuilderConfig* config{ nullptr };
    nvinfer1::IHostMemory* modelStream{ nullptr };
    nvinfer1::ICudaEngine* engine{ nullptr };
    nvinfer1::IExecutionContext* context{ nullptr };
    bool isDynamic{true};//模型是否是动态batch
    int INFERENCE_BATCH{-1};
    std::vector<int> inputdims; std::vector<int> outputdims;
    int inputLen{1}; int outputLen{1};//不包含batchsize
    // 不同平台下文件夹搜索工具
    std::map<std::string, int> class2label;

    void doInference(nvinfer1::IExecutionContext&context,  float* input, float* output, int batchsize);

};

#endif // TRTINFER_H
