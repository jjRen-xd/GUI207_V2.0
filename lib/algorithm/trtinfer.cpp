#include <torch/torch.h>
#include "trtinfer.h"
#include <QDebug>

#include <Windows.h> //for sleep
using namespace nvinfer1;
static Logger gLogger;

TrtInfer::TrtInfer(std::map<std::string, int> class2label):class2label(class2label){

}
void getDataFromMat(std::string targetMatFile,int emIdx,float *data){
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
    for(int i=0;i<N;i++){
        data[i]=matdata[M*i+emIdx];
    }
//    mxFree(pMxArray);
//    matClose(pMatFile);//不注释这两个善后代码就会crashed，可能是冲突了
}
bool readTRTFile(const std::string& engineFile, IHostMemory*& trtModelStream, ICudaEngine*& engine)
{
    std::fstream file;
    std::cout << "(TrtInfer:read_TRT_File)loading filename from:" << engineFile << std::endl;
    nvinfer1::IRuntime* trtRuntime;
    //nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
    std::cout << "AAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
    file.open(engineFile, std::ios::binary | std::ios::in);
    std::cout << "BBBBBBBBBBBBBBBBBBBBBBB" << std::endl;
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

class dataSetClc: public torch::data::Dataset<dataSetClc>{
public:
    int class_index = 0;
    dataSetClc(std::string data_dir, std::string type, std::map<std::string, int> class2label):
        class2label(class2label){
        load_data_from_folder(data_dir, type, dataPaths, labels, class2label);
    }

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override{
        std::string data_path = dataPaths.at(index);
        torch::Tensor data_tensor = getTensorFromTXT(data_path);
        int label = labels.at(index);
        torch::Tensor label_tensor = torch::full({1}, label, torch::kInt64);
        return {data_tensor.clone(), label_tensor.clone()};
    }

    // Override size() function, return the length of data
    torch::optional<size_t> size() const override{
        return dataPaths.size();
    };

private:
    std::vector<std::string> dataPaths;
    std::vector<int> labels;
    std::map<std::string, int> class2label;
};

void TrtInfer::doInference(IExecutionContext&context, float* input, float* output){
    void* buffers[2] = { NULL,NULL };
    cudaMalloc(&buffers[0], batchSize * input_len * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * output_len * sizeof(float));

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(buffers[0], input, batchSize * input_len * sizeof(float), cudaMemcpyHostToDevice, stream);
    //start to infer
    std::cout << "Start to infer ..." << std::endl;
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], batchSize * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    std::cout << "Inference Done." << std::endl;
}

void TrtInfer::testOneSample(std::string targetPath, int emIndex, std::string modelPath, int &predIdx,std::vector<float> &degrees){
    //qDebug() << "(OnnxInfer::testOneSample)子线程id：" << QThread::currentThreadId();
    if (readTRTFile(modelPath,modelStream, engine)) std::cout << "(TrtInfer::testOneSample)tensorRT engine created successfully." << std::endl;
    else std::cout << "(TrtInfer::testOneSample)tensorRT engine created failed." << std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //根据模型确认输入输出的数据尺度
    input_len=1;output_len=1;
    nvinfer1::Dims indims = engine->getBindingDimensions(0);
    for (int i = 0; i < indims.nbDims; i++){
        if(i!=0) input_len*=indims.d[i];
        inputdims.push_back(indims.d[i]);
    }
    nvinfer1::Dims oudims = engine->getBindingDimensions(1);
    for (int i = 0; i < oudims.nbDims; i++){
        if(i!=0) output_len*=oudims.d[i];
        outputdims.push_back(oudims.d[i]);
    }
    if(inputdims[0]==outputdims[0]&&inputdims[0]==-1) isDynamic=true;
    else if(inputdims[0]==outputdims[0]) batchSize=inputdims[0];
    else {qDebug()<<"模型输入输出批数不一致！";return;}

    qDebug()<<"================打印当前trt模型的输入输出维度================";//默认是按两个dims写了，只有一个输入一个输出。
    qDebug()<<"inputdims:(";
    for (int i = 0; i < indims.nbDims; i++) qDebug()<<inputdims[i]<<",";
    qDebug()<<"outputdims:(";
    for (int i = 0; i < oudims.nbDims; i++) qDebug()<<outputdims[i]<<",";
    qDebug()<<"=========================================================";

    float *indata=new float[input_len]; std::fill_n(indata,input_len,0);
    float *outdata=new float[output_len]; std::fill_n(outdata,output_len,0);
    getDataFromMat(targetPath,emIndex,indata);

    batchSize=1;
    doInference(*context, indata, outdata);

    torch::Tensor output_tensor = torch::ones({1,output_len});
    std::cout << "(TrtInfer::testOneSample)output_tensor:  ";
    for (unsigned int i = 0; i < output_len; i++){
        std::cout << outdata[i] << ", ";
        output_tensor[0][i]=outdata[i];
    }
    output_tensor = torch::softmax(output_tensor, 1).flatten();
    int pred=output_tensor.argmax(0).item<int>();
    //int classnum=5;
    qDebug()<< "(TrtInfer::testOneSample)predicted label:"<<QString::number(pred);
    std::vector<float> output(output_tensor.data_ptr<float>(),output_tensor.data_ptr<float>()+output_tensor.numel());

    //std::cout << "(OnnxInfer::testOneSample)output[2]="<<output[0]<<std::endl;
    //for(int i=0;i<output.size();i++) std::cout<<output[i]<<" ";
    //std::cout << "(OnnxInfer::testOneSample)Here~"<<std::endl;
    degrees=output;

    //emit finished(pred);
    predIdx=pred;
    //qDebug() << "subThread Done! pred=" << pred;
}

//void TrtInfer::testAllSample(std::string dataset_path,std::string model_path,float &Acc,std::vector<std::vector<int>> &confusion_matrix){
//    if (read_TRT_File(model_path,modelStream, engine)) std::cout << "tensorRT engine created successfully." << std::endl;
//    else std::cout << "tensorRT engine created failed." << std::endl;
//    context = engine->createExecutionContext();
//    assert(context != nullptr);



//    // LOAD DataSet
//    auto test_dataset = dataSetClc(dataset_path, ".txt", class2label)
//            .map(torch::data::transforms::Stack<>());
//    const size_t test_dataset_size = test_dataset.size().value();

//    // batch : data_loader数据量为1
//    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 1);

//    int correct=0,real=0,guess=0;
//    float data_f[512] = { 0 };
//    float prob_f[5] = { 0 };
//    torch::Tensor output_tensor = torch::ones({1,5});
//    for (const auto &batch : *test_loader){
//        auto data_tensor = batch.data;
//        auto targets_tensor = batch.target;
//        targets_tensor.resize_({1});
//        auto res = data_tensor.accessor<float,3>();            //拙劣待优化的tensor转float[]
//        for(int i=0;i<512;i++){
//            data_f[i]=res[0][0][i];
//        }
//        doInference(*context, data_f, prob_f, 1);
//        for (unsigned int i = 0; i < 5; i++){
//            std::cout << prob_f[i] << ", ";
//            output_tensor[0][i]=prob_f[i];
//        }
//        auto pred = output_tensor.argmax(1);
//        correct += pred.eq(targets_tensor).sum().template item<int64_t>();
//        real=targets_tensor[0].item<int>();
//        guess=output_tensor.view(-1).argmax().item<int>();
//        confusion_matrix[real][guess]++;
//    }

//    std::cout << "correct:"<<correct<<std::endl;
//    std::cout << "test_dataset_size:"<<test_dataset_size<<std::endl;
//    Acc=static_cast<float> (correct) / (test_dataset_size);
//}

