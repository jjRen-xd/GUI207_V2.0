#include <torch/torch.h>
#include "trtinfer.h"
#include <QDebug>

#include <Windows.h> //for sleep
using namespace nvinfer1;
static Logger gLogger;

TrtInfer::TrtInfer(std::map<std::string, int> class2label):class2label(class2label){

}

void getAllDataFromMat(std::string matPath,std::vector<torch::Tensor> &data,std::vector<int> &labels,int label,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(trtInfer:getDataFromMat)文件指针空！！！！！！";
        return;
    }
    std::string matVariable="hrrp128";//假设数据变量名同文件名的话就filefullpath.split(".").last().toStdString().c_str()
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(trtInfer:getDataFromMat)pMxArray变量没找到！！！！！！";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //M=36 样本数量
    int N = mxGetN(pMxArray);  //N=128 单一样本维度
    for(int i=0;i<M;i++){
        torch::Tensor temp=torch::rand({inputLen});
        for(int j=0;j<inputLen;j++){
            temp[j]=matdata[M*(j%N)+i];//如果inputLen比N还小，不会报错，但显然数据集和模型是不对应的吧，得到的推理结果应会很难看
        }
        //std::cout<<&temp<<std::endl;
        data.push_back(temp);
        labels.push_back(label);
    }
}

void loadAllDataFromFolder(std::string datasetPath,std::string type,std::vector<torch::Tensor> &data,
                           std::vector<int> &labels,std::map<std::string, int> &class2label,int inputLen){
    SearchFolder *dirTools = new SearchFolder();
    // 寻找子文件夹 WARN:数据集的路径一定不能包含汉字 否则遍历不到文件路径
    std::vector<std::string> subDirs;
    dirTools->getDirs(subDirs, datasetPath);
    for(auto &subDir: subDirs){
        // 寻找每个子文件夹下的样本文件
        std::vector<std::string> fileNames;
        std::string subDirPath = datasetPath+"/"+subDir;
        dirTools->getFiles(fileNames, type, subDirPath);
        for(auto &fileName: fileNames){
            //qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName)<<" label:"<<class2label[subDir];
            getAllDataFromMat(subDirPath+"/"+fileName,data,labels,class2label[subDir],inputLen);
        }
    }
    return;
}

class CustomDataset: public torch::data::Dataset<CustomDataset>{
private:
    std::vector<torch::Tensor> data;
    std::vector<int> labels;
    std::map<std::string, int> class2label;
    int inputLen;
public:
    CustomDataset(std::string dataSetPath, std::string type, std::map<std::string, int> class2label,int inputLen)
        :class2label(class2label),inputLen(inputLen){
        loadAllDataFromFolder(dataSetPath, type, data, labels, class2label,inputLen);
//        for(int i=0;i<5;i++){
//            torch::Tensor data_tensor = data.at(i);
//            data_tensor=data_tensor.flatten();
//            std::cout<<"data["<<i<<"]:"<< "data_tensor.sizes()="<< data_tensor.sizes()<<"  data_tensor.numel()="<< data_tensor.numel() << std::endl;
//        }
    }

    torch::data::Example<> get(size_t index) override{
        torch::Tensor data_tensor = data.at(index);
        int label = labels.at(index);
        torch::Tensor label_tensor = torch::full({1}, label, torch::kInt64);
        return {data_tensor.clone(), label_tensor.clone()};
    }

    // Override size() function, return the length of data
    torch::optional<size_t> size() const override{
        return labels.size();
    };
};

void getDataFromMat(std::string targetMatFile,int emIdx,float *data,int input_len){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(targetMatFile.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(trtInfer:getDataFromMat)文件指针空！！！！！！";
        return;
    }
    std::string matVariable="hrrp128";//filefullpath.split(".").last().toStdString().c_str() 假设数据变量名同文件名的话
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(trtInfer:getDataFromMat)pMxArray变量没找到！！！！！！";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //M=36 样本数量
    int N = mxGetN(pMxArray);  //N=128 单一样本维度
    if(emIdx>M) emIdx=M-1; //说明是随机数
    for(int i=0;i<input_len;i++){
        data[i]=matdata[M*(i%N)+emIdx];//一个
    }
//    mxFree(pMxArray);
//    matClose(pMatFile);//不注释这两个善后代码就会crashed，可能是冲突了
}

bool readTrtFile(const std::string& engineFile, IHostMemory*& trtModelStream, ICudaEngine*& engine){
    std::fstream file;
    std::cout << "(TrtInfer:read_TRT_File)loading filename from:" << engineFile << std::endl;
    nvinfer1::IRuntime* trtRuntime;
    //nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
    file.open(engineFile, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    //std::cout << "length:" << length << std::endl;
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    std::cout << "(TrtInfer:read_TRT_File)load engine done" << std::endl;
    std::cout << "(TrtInfer:read_TRT_File)deserializing" << std::endl;
    trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    //ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
    engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    std::cout << "(TrtInfer:read_TRT_File)deserialize done" << std::endl;
    assert(engine != nullptr);
    std::cout << "(TrtInfer:read_TRT_File)Great. The engine in TensorRT.cpp is not nullptr" << std::endl;
    trtModelStream = engine->serialize();
    return true;
}

void TrtInfer::doInference(IExecutionContext&context, float* input, float* output, int batchsize){
    void* buffers[2] = { NULL,NULL };
    cudaMalloc(&buffers[0], batchsize * inputLen * sizeof(float));
    cudaMalloc(&buffers[1], batchsize * outputLen * sizeof(float));

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(buffers[0], input, batchsize * inputLen * sizeof(float), cudaMemcpyHostToDevice, stream);
    //start to infer
    //qDebug()<< "Start to infer ..." ;
    context.enqueue(batchsize, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], batchsize * outputLen * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    //qDebug()<< "Inference Done." ;
}

void TrtInfer::testOneSample(std::string targetPath, int emIndex, std::string modelPath, int *predIdx,std::vector<float> &degrees){
    //qDebug() << "(OnnxInfer::testOneSample)子线程id：" << QThread::currentThreadId();
    if (readTrtFile(modelPath,modelStream, engine)) qDebug()<< "(TrtInfer::testOneSample)tensorRT engine created successfully." ;
    else qDebug()<< "(TrtInfer::testOneSample)tensorRT engine created failed." ;
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //根据模型确认输入输出的数据尺度
    inputLen=1;outputLen=1;
    nvinfer1::Dims indims = engine->getBindingDimensions(0);
    for (int i = 0; i < indims.nbDims; i++){
        if(i!=0) inputLen*=indims.d[i];
        inputdims.push_back(indims.d[i]);
    }
    nvinfer1::Dims oudims = engine->getBindingDimensions(1);
    for (int i = 0; i < oudims.nbDims; i++){
        if(i!=0) outputLen*=oudims.d[i];
        outputdims.push_back(oudims.d[i]);
    }
    if(inputdims[0]==outputdims[0]&&inputdims[0]==-1) isDynamic=true;
    else if(inputdims[0]==outputdims[0]) INFERENCE_BATCH=inputdims[0];
    else {qDebug()<<"模型输入输出批数不一致！";return;}

    /*qDebug()<<"================打印当前trt模型的输入输出维度================";//默认是按两个dims写了，只有一个输入一个输出。
    qDebug()<<"inputdims:(";
    for (int i = 0; i < indims.nbDims; i++) qDebug()<<inputdims[i]<<",";
    qDebug()<<"outputdims:(";
    for (int i = 0; i < oudims.nbDims; i++) qDebug()<<outputdims[i]<<",";
    qDebug()<<"=========================================================";*/

    INFERENCE_BATCH=1;
    //define the dims if necessary
    if(isDynamic){
        nvinfer1::Dims dims4;   dims4.d[0]=INFERENCE_BATCH;
        for(int i=1;i<indims.nbDims;i++) dims4.d[i]=inputdims[i];
        dims4.nbDims = indims.nbDims;
        context->setBindingDimensions(0, dims4);
    }
    //ready to send data to context
    float *indata=new float[inputLen]; std::fill_n(indata,inputLen,0);
    float *outdata=new float[outputLen]; std::fill_n(outdata,outputLen,0);
    getDataFromMat(targetPath,emIndex,indata,inputLen);

    //qDebug()<<"indata[]_len=="<<QString::number(inputLen)<<"   outdata[]_len=="<<QString::number(outputLen);
    doInference(*context, indata, outdata, 1);

    torch::Tensor output_tensor = torch::ones({outputLen});
    std::cout << "(TrtInfer::testOneSample)output_tensor:  ";
    for (unsigned int i = 0; i < outputLen; i++){
        std::cout << outdata[i] << ", ";
        output_tensor[i]=outdata[i];
    }
    //output_tensor = torch::softmax(output_tensor, 0).flatten();
    int pred=output_tensor.argmax(0).item<int>();
    //int classnum=5;
    qDebug()<< "(TrtInfer::testOneSample)predicted label:"<<QString::number(pred);
    std::vector<float> output(output_tensor.data_ptr<float>(),output_tensor.data_ptr<float>()+output_tensor.numel());
    degrees=output;
    *predIdx=pred;
}

void TrtInfer::testAllSample(std::string dataset_path,std::string modelPath,float &Acc,std::vector<std::vector<int>> &confusion_matrix){
    if (readTrtFile(modelPath,modelStream, engine)) qDebug()<< "(TrtInfer::testAllSample)tensorRT engine created successfully.";
    else qDebug()<< "(TrtInfer::testAllSample)tensorRT engine created failed." ;
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //根据模型确认输入输出的数据尺度
    inputLen=1;outputLen=1;
    nvinfer1::Dims indims = engine->getBindingDimensions(0);
    for (int i = 0; i < indims.nbDims; i++){
        if(i!=0) inputLen*=indims.d[i];
        inputdims.push_back(indims.d[i]);
    }
    nvinfer1::Dims oudims = engine->getBindingDimensions(1);
    for (int i = 0; i < oudims.nbDims; i++){
        if(i!=0) outputLen*=oudims.d[i];
        outputdims.push_back(oudims.d[i]);
    }
    if(inputdims[0]==outputdims[0]&&inputdims[0]==-1) isDynamic=true;
    else if(inputdims[0]==outputdims[0]) INFERENCE_BATCH=inputdims[0];
    else {qDebug()<<"模型输入输出批数不一致！";return;}
    ///如果isDynamic=TRUE, 应使提供设置batch的选项可选，同时把maxBatch传过去
    INFERENCE_BATCH=INFERENCE_BATCH==-1?1:INFERENCE_BATCH;//so you should specific Batch before this line

    if(isDynamic){
        nvinfer1::Dims dims4;   dims4.d[0]=INFERENCE_BATCH;
        for(int i=1;i<indims.nbDims;i++) dims4.d[i]=inputdims[i];
        dims4.nbDims = indims.nbDims;
        context->setBindingDimensions(0, dims4);
    }

    qDebug()<<"(TrtInfer::testAllSample) INFERENCE_BATCH==="<<INFERENCE_BATCH;
    qDebug()<<"(TrtInfer::testAllSample) inputLen==="<<inputLen;
    // LOAD DataSet
    auto test_dataset = CustomDataset(dataset_path, ".mat", class2label,inputLen)
            .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), INFERENCE_BATCH);

    //auto test_dataset_size = test_dataset.size().value();
    //std::cout<<"(TrtInfer::testAllSample) test_dataset_size"<<test_dataset_size<<std::endl;

    int correct=0,real=0,guess=0;
    qDebug()<<"(TrtInfer::testAllSample) AAAAAAAAAAAA";
    int test_dataset_size=0;
    for (const auto &batch : *test_loader){
        auto indata_tensor = batch.data;
        auto labels_tensor = batch.target;
        int thisBatch=labels_tensor.numel();
        float *indata=new float[thisBatch*inputLen]; std::fill_n(indata,inputLen,6);
        float *outdata=new float[thisBatch*outputLen]; std::fill_n(outdata,outputLen,6);
        torch::Tensor output_tensor = torch::ones({thisBatch,outputLen});

        //targets_tensor.resize_({INFERENCE_BATCH});
        indata_tensor=indata_tensor.flatten();
        labels_tensor=labels_tensor.flatten();

//        auto a_size = indata_tensor.sizes();
//        int num_ = indata_tensor.numel();
//        std::cout << "======indata_tensor:" << a_size << "    &&&     " << num_ << std::endl;
//        auto a_size2 = labels_tensor.sizes();
//        int num_2 = labels_tensor.numel();
//        std::cout << "======targets_tensor:" << a_size2 << "    &&&     " << num_2 << std::endl;

        auto res = indata_tensor.accessor<float,1>();            //拙劣待优化的tensor转float[]
        memcpy(indata, res.data(), inputLen*thisBatch*sizeof(float));

        doInference(*context, indata, outdata, thisBatch);

//        std::cout<<"(TrtInfer::testAllSample) after inference outdata:"<<std::endl;
//        for (unsigned int i = 0; i < outputLen*INFERENCE_BATCH; i++){
//            std::cout << outdata[i] << ", ";
//            output_tensor[i/outputLen][i%outputLen]=outdata[i];
//        }std::cout<<std::endl;

        auto pred = output_tensor.argmax(1);
        correct += pred.eq(labels_tensor).sum().template item<int64_t>();
        for(int i=0;i<thisBatch;i++){
            real=labels_tensor[i].item<int>();
            guess=pred[i].item<int>();
            confusion_matrix[real][guess]++;
            test_dataset_size++;
        }

    }
    qDebug()<<"test_dataset_size="<<test_dataset_size;
    std::cout << "correct:"<<correct<<std::endl;
    //std::cout << "test_dataset_size:"<<test_dataset_size<<std::endl;
    Acc=test_dataset_size==0?0:static_cast<float> (correct) / (test_dataset_size);
}

void TrtInfer::setBatchSize(int batchSize){
    INFERENCE_BATCH=batchSize;
}
