#ifndef ONNXINFER_H
#define ONNXINFER_H

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
class OnnxInfer : public QObject
{
    Q_OBJECT
public:
    OnnxInfer(std::map<std::string, int> class2label);

public slots:
    void doInference(nvinfer1::IExecutionContext&context, float* input, float* output, int batchSize);
    void testOneSample(std::string targetPath, std::string modelPath, std::promise<int> *predIdx, std::promise<std::vector<float>> *degrees);
    void testOneSample2(std::string targetPath, std::string modelPath, std::promise<int>  *predIdx, std::promise<std::vector<float>> *degrees);
    void testAllSample(std::string hrrpdataset_path,std::string hrrpmodel_path,float &Acc,std::vector<std::vector<int>> &confusion_matrix);
signals:
    void finished(int pred);
private:
    //bool read_TRT_File(const std::string& engineFile, nvinfer1::IHostMemory*& trtModelStream, nvinfer1::ICudaEngine*& engine);
    nvinfer1::IBuilder* builder{ nullptr };
    nvinfer1::INetworkDefinition* network{ nullptr };
    nvinfer1::IBuilderConfig* config{ nullptr };
    nvinfer1::IHostMemory* modelStream{ nullptr };
    nvinfer1::ICudaEngine* engine{ nullptr };
    nvinfer1::IExecutionContext* context{ nullptr };
    //void getFloatFromTXT(std::string data_path,float* y);
    //BashTerminal *terminal;
    // 不同平台下文件夹搜索工具
    std::map<std::string, int> class2label;
protected:

};

#endif // ONNXINFER_H
