#include <torch/torch.h>
#include "trtinfer.h"
#include <QDebug>

#include <Windows.h> //for sleep
using namespace nvinfer1;
static Logger gLogger;

TrtInfer::TrtInfer(std::map<std::string, int> class2label):class2label(class2label){

}

void oneNormalization(std::vector<float> &list){
    //特征归一化
    float dMaxValue = *max_element(list.begin(),list.end());  //求最大值
    //std::cout<<"maxdata"<<dMaxValue<<'\n';
    float dMinValue = *min_element(list.begin(),list.end());  //求最小值
    //std::cout<<"mindata"<<dMinValue<<'\n';
    for (int f = 0; f < list.size(); ++f) {
        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//极小值限制
    }
}

void getAllDataFromMat(std::string matPath,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen){
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
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    for(int i=0;i<N;i++){
        std::vector<float> onesmp;//存当前遍历的一个样本
        for(int j=0;j<M;j++){
            onesmp.push_back(matdata[i*M+j]);
        }
        oneNormalization(onesmp);//归一化
        std::vector<float> temp;
        for(int j=0;j<inputLen;j++){
            temp.push_back(onesmp[j%M]);//如果inputLen比N还小，不会报错，但显然数据集和模型是不对应的吧，得到的推理结果应会很难看
        }
        //std::cout<<&temp<<std::endl;
        data.push_back(temp);
        labels.push_back(label);
    }
}

void loadAllDataFromFolder(std::string datasetPath,std::string type,std::vector<std::vector<float>> &data,
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

class CustomDataset{
public:
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    std::map<std::string, int> class2label;
    CustomDataset(std::string dataSetPath, std::string type, std::map<std::string, int> class2label,int inputLen)
        :class2label(class2label){
        loadAllDataFromFolder(dataSetPath, type, data, labels, class2label,inputLen);

    }
    int size(){
        return labels.size();
    };
};

void getDataFromMat(std::string targetMatFile,int emIdx,float *data,int inputLen){
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
    int M = mxGetM(pMxArray);  //M=128 行数
    int N = mxGetN(pMxArray);  //N=1000 列数
    if(emIdx>N) emIdx=N-1; //说明是随机数

    std::vector<float> onesmp;//存当前样本
    for(int i=0;i<M;i++){
        onesmp.push_back(matdata[emIdx*M+i]);
    }
    oneNormalization(onesmp);//归一化
    for(int i=0;i<inputLen;i++){
        data[i]=onesmp[i%M];//matlab按列存储
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
    context.enqueueV2(buffers, stream, nullptr);
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
    for(int i=0;i<inputLen;i++) std::cout<<indata[i]<<" ";std::cout<<std::endl;
    for(int i=0;i<outputLen;i++) std::cout<<outdata[i]<<" ";std::cout<<std::endl;
    //qDebug()<<"indata[]_len=="<<QString::number(inputLen)<<"   outdata[]_len=="<<QString::number(outputLen);
    doInference(*context, indata, outdata, INFERENCE_BATCH);
    std::cout<<"======================================================"<<std::endl;
    for(int i=0;i<inputLen;i++) std::cout<<indata[i]<<" ";std::cout<<std::endl;
    for(int i=0;i<outputLen;i++) std::cout<<outdata[i]<<" ";std::cout<<std::endl;
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
    INFERENCE_BATCH=3;
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
    clock_t start,end;
    start = clock();


    auto test_dataset = CustomDataset(dataset_path, ".mat", class2label,inputLen);

//    auto test_dataset = CustomDataset(dataset_path, ".mat", class2label,inputLen);
//    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), INFERENCE_BATCH);

    end = clock();

    int correct=0,real=0,guess=0;
    qDebug()<<"(TrtInfer::testAllSample) DataLoader Check.";
    int test_dataset_size=test_dataset.labels.size();
    //qDebug()<<"(TrtInfer::testAllSample) DataSetsize: "<<test_dataset.data.size()<<"LabelSize: "<<test_dataset.labels.size();
    qDebug()<<"(TrtInfer::testAllSample) 数据加载用时: "<<(double)(end-start)/CLOCKS_PER_SEC;


    for(int l=1;l<=ceil(test_dataset_size/INFERENCE_BATCH);l++){            //分批喂数据
        std::vector<float> thisBatchData;
        std::vector<float> thisBatchLabels;
        int thisBatchNum=1;
        if(l==ceil(test_dataset_size/INFERENCE_BATCH)) thisBatchNum=test_dataset_size-(l-1)*INFERENCE_BATCH;
        else thisBatchNum=INFERENCE_BATCH;

        int beginIdx=(l-1)*INFERENCE_BATCH;
        for(int i=0;i<thisBatchNum;i++){
            thisBatchData.insert(thisBatchData.end(),test_dataset.data[beginIdx+i].begin(),test_dataset.data[beginIdx+i].end());
            thisBatchLabels.push_back(test_dataset.labels[beginIdx+i]);
        }

        float *indata=new float[thisBatchNum*inputLen]; std::fill_n(indata,inputLen,class2label.size());
        float *outdata=new float[thisBatchNum*outputLen]; std::fill_n(outdata,outputLen,class2label.size());
        if (!thisBatchData.empty()){
            memcpy(indata, &thisBatchData[0], thisBatchData.size()*sizeof(float));
        }
//        for(int i=0;i<thisBatchNum*inputLen;i++){
//            std::cout<<indata[i]<<" ";
//        }return;
//        auto a_size = indata_tensor.sizes();
//        int num_ = indata_tensor.numel();
//        std::cout << "======indata_tensor:" << a_size << "    &&&     " << num_ << std::endl;
//        auto a_size2 = labels_tensor.sizes();
//        int num_2 = labels_tensor.numel();
//        std::cout << "======targets_tensor:" << a_size2 << "    &&&     " << num_2 << std::endl;


        doInference(*context, indata, outdata, thisBatchNum);

//        std::cout<<"(TrtInfer::testAllSample) after inference outdata:"<<std::endl;
        //torch::Tensor output_tensor = torch::ones({thisBatchNum,outputLen});
        std::vector<std::vector<float>> output_vec;
        std::vector<float> temp;
        for (int i = 1; i <= outputLen*thisBatchNum; i++){
            //std::cout << outdata[i-1] << ", ";
            //output_tensor[i/outputLen][i%outputLen]=outdata[i];
            temp.push_back(outdata[i-1]);
            if(i%outputLen==0){
                output_vec.push_back(temp);
                temp.clear();
            }
        }//std::cout<<std::endl;

        //auto pred = output_tensor.argmax(1);
        //std::cout<<std::endl<<"pred.sizes()="<<pred.sizes()<<"pred.numel()="<<pred.numel()<<std::endl;

        for(int i=0;i<thisBatchNum;i++){
            int guess=max_element(output_vec[i].begin(), output_vec[i].end())-output_vec[i].begin();
            int real=test_dataset.labels[beginIdx+i];
            if(guess==real) correct ++;
            confusion_matrix[real][guess]++;
            //std::cout<<"confusion_matrix["<<real<<"]["<<guess<<"]++"<<std::endl;
        }


    }
    qDebug()<<"test_dataset_size="<<test_dataset_size;
    qDebug()<< "correct:"<<correct;

    Acc=test_dataset_size==0?0:static_cast<float> (correct) / (test_dataset_size);
}

void TrtInfer::setBatchSize(int batchSize){
    INFERENCE_BATCH=batchSize;
}
