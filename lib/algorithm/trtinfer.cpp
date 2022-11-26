//#include <torch/torch.h>
#include "trtinfer.h"
#include <QDebug>
#include <QMessageBox>
#include <Windows.h> //for sleep
using namespace nvinfer1;
static Logger gLogger;

TrtInfer::TrtInfer(std::map<std::string, int> class2label):class2label(class2label){

}



bool readTrtFile(const std::string& engineFile, IHostMemory*& trtModelStream, ICudaEngine*& engine){
    std::fstream file;
    //std::cout << "(TrtInfer:read_TRT_File)loading filename from:" << engineFile << std::endl;
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
    //std::cout << "(TrtInfer:read_TRT_File)load engine done" << std::endl;
    //std::cout << "(TrtInfer:read_TRT_File)deserializing" << std::endl;
    trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    //ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
    engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    //std::cout << "(TrtInfer:read_TRT_File)deserialize done" << std::endl;
    assert(engine != nullptr);
    //std::cout << "(TrtInfer:read_TRT_File)Great. The engine in TensorRT.cpp is not nullptr" << std::endl;
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
    //qDebug()<< "Start to infer ..." ;
//    context.enqueue(batchsize, buffers, stream, nullptr);
    context.enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], batchsize * outputLen * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    //qDebug()<< "Inference Done." ;
}

QString TrtInfer::testOneSample(
        std::string targetPath, int emIndex, std::string modelPath, bool dataProcess,
        int *predIdx,std::vector<float> &degrees,std::string flag)
{
    qDebug()<<"(TrtInfer::testOneSample)modelPath="<<QString::fromStdString(modelPath);
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
    else {qDebug()<<"模型输入输出批数不一致！";return QString::fromStdString("error");}

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
    if(flag=="FEA_RELE_abfc"){
        std::string theClassPath=targetPath.substr(0,targetPath.rfind("/"));
        std::string theClass=theClassPath.substr(theClassPath.rfind("/")+1,theClassPath.size());
        std::string dataset_path=theClassPath.substr(0,theClassPath.rfind("/"));

        CustomDataset test_dataset_for_abfc = CustomDataset(dataset_path,dataProcess, ".mat", class2label, inputLen ,flag, modelIdx, dataOrder);
        test_dataset_for_abfc.getDataSpecifically(theClass,emIndex,indata);

    }
    else{
        MatDataProcess matDataPrcs;
        matDataPrcs.getDataFromMat(targetPath,emIndex,dataProcess,indata,inputLen);
    }
    // std::cout<<"(TrtInfer::testOneSample) print data[0]:"<<std::endl;
    //for(int i=0;i<128;i++) std::cout<<indata[i]<<" ";std::cout<<std::endl;

    qDebug()<<"(TrtInfer::testOneSample)indata[]_len=="<<QString::number(inputLen)<<"   outdata[]_len=="<<QString::number(outputLen);
    clock_t start,end;
    start = clock();
    doInference(*context, indata, outdata, 1);
    end = clock();
    QString inferTime=QString::number((double)(end-start)/CLOCKS_PER_SEC);//s
    std::vector<float> output_vec;
    std::cout << "(TrtInfer::testOneSample)outdata:  ";
    float outdatasum=0.0;
    for (unsigned int i = 0; i < outputLen; i++){
        std::cout << outdata[i] << ", ";
        outdatasum+=outdata[i];
        output_vec.push_back(outdata[i]);
    }std::cout<<std::endl;

    //和不为1说明网络模型最后一层不是softmax，就主动做一下softmax
    if(abs(outdatasum-1.0)>1e-8) softmax(output_vec);

    int pred = std::distance(output_vec.begin(),std::max_element(output_vec.begin(),output_vec.end()));
    qDebug()<< "(TrtInfer::testOneSample)predicted label:"<<QString::number(pred);
    degrees=output_vec;
    *predIdx=pred;
    return inferTime;
}

bool TrtInfer::testAllSample(
        std::string dataset_path,std::string modelPath,int inferBatch, bool dataProcess,
        float &Acc,std::vector<std::vector<int>> &confusion_matrix,std::string flag){
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
        //qDebug()<<"indims[]"<<indims.d[i];
    }
    nvinfer1::Dims oudims = engine->getBindingDimensions(1);
    for (int i = 0; i < oudims.nbDims; i++){
        if(i!=0) outputLen*=oudims.d[i];
        outputdims.push_back(oudims.d[i]);
    }
    if(inputdims[0]==outputdims[0]&&inputdims[0]==-1) isDynamic=true;
    else if(inputdims[0]==outputdims[0]) INFERENCE_BATCH=inputdims[0];
    else {qDebug()<<"模型输入输出批数不一致！";return 0;}
    ///如果isDynamic=TRUE, 应使提供设置batch的选项可选，同时把maxBatch传过去
    INFERENCE_BATCH=1;//这里是写死了，应该传过来
    INFERENCE_BATCH=INFERENCE_BATCH==-1?1:INFERENCE_BATCH;//so you should specific Batch before this line

    if(isDynamic){
        nvinfer1::Dims dims4;   dims4.d[0]=INFERENCE_BATCH;
        for(int i=1;i<indims.nbDims;i++) dims4.d[i]=inputdims[i];
        dims4.nbDims = indims.nbDims;
        try{
            context->setBindingDimensions(0, dims4);
        }
        catch (...) {
            QMessageBox::information(NULL, "所有样本测试", "批处理量超过模型预设值！");
            return 0;
        }
    }
    //qDebug()<<"(TrtInfer::testAllSample) INFERENCE_BATCH==="<<INFERENCE_BATCH;
    qDebug()<<"(TrtInfer::testAllSample) modelInputLen==="<<inputLen;

    // LOAD DataSet
    clock_t start,end;
    start = clock();
    CustomDataset test_dataset = CustomDataset(dataset_path,dataProcess, ".mat", class2label, inputLen ,flag, modelIdx, dataOrder);

    qDebug()<<"(TrtInfer::testAllSample) test_dataset.data.size()==="<<test_dataset.data.size();
    //qDebug()<<"(TrtInfer::testAllSample) test_dataset.label.size()==="<<test_dataset.labels.size();
    end = clock();
    int correct=0;
    //qDebug()<<"(TrtInfer::testAllSample) DataLoader Check.";
    int test_dataset_size=test_dataset.labels.size();
    //qDebug()<<"(TrtInfer::testAllSample) DataSetsize: "<<test_dataset.data.size()<<"LabelSize: "<<test_dataset.labels.size();
    qDebug()<<"(TrtInfer::testAllSample) 数据加载用时: "<<(double)(end-start)/CLOCKS_PER_SEC;
    float test_dataset_size_float=test_dataset_size;
    for(int l=1;l<=ceil(test_dataset_size_float/INFERENCE_BATCH);l++){            //分批喂数据
        std::vector<float> thisBatchData;
        std::vector<float> thisBatchLabels;
        int thisBatchNum=1;//这批数据的number
        if(l==ceil(test_dataset_size_float/INFERENCE_BATCH)) thisBatchNum=test_dataset_size-(l-1)*INFERENCE_BATCH;
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
        // qDebug()<<"(TrtInfer::testAllSample) beginIdx="<<QString::number(beginIdx)<<"  label=="<<QString::number(test_dataset.labels[beginIdx]);
        // if(beginIdx==50){
        //     std::cout<<"(TrtInfer::testAllSample) print first col of DT:";
        //     for(int i=0;i<128;i++) std::cout<<indata[i]<<" ";
        //     std::cout<<std::endl;
        // }
        doInference(*context, indata, outdata, thisBatchNum);

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
    // std::cout<<std::endl<<"打印源混淆矩阵"<<std::endl;
    // for(int i=0;i<confusion_matrix.size();i++){
    //     for(int j=0;j<confusion_matrix[i].size();j++){
    //         std::cout<<confusion_matrix[i][j]<<" ";
    //     }std::cout<<std::endl;
    // }
    qDebug()<<"test_dataset_size="<<test_dataset_size;
    qDebug()<< "correct:"<<correct;

    Acc=test_dataset_size==0?0:static_cast<float> (correct) / (test_dataset_size);
    return 1;
}

void TrtInfer::realTimeInfer(std::vector<float> data_vec,std::string modelPath, bool dataProcess,int *predIdx, std::vector<float> &degrees){

    //int inputLen=2;int outputLen=6;
    //ready to send data to context
    if(dataProcess) oneNormalization_(data_vec);//对收到的数据做归一
    float *outdata=new float[outputLen]; std::fill_n(outdata,outputLen,9);
    //qDebug()<<"(TrtInfer::realTimeInfer) i get one! now to infer";
    float *indata = new float[data_vec.size()];
    if (!data_vec.empty()){
        memcpy(indata, &data_vec[0], data_vec.size()*sizeof(float));
    }
    //qDebug()<<"(TrtInfer::realTimeInfer)  indata[0]==="<<indata[0];
    doInference(*context, indata, outdata, 1);
    std::vector<float> output_vec;
    //std::cout << "(TrtInfer::testOneSample)outdata:  ";
    float outdatasum=0.0;
    for (unsigned int i = 0; i < outputLen; i++){
        //std::cout << outdata[i] << ", ";
        outdatasum+=outdata[i];
        output_vec.push_back(outdata[i]);
    }//std::cout<<std::endl;

    //和不为1说明网络模型最后一层不是softmax，就主动做一下softmax
    if(abs(outdatasum-1.0)>1e-8) softmax(output_vec);

    int pred = std::distance(output_vec.begin(),std::max_element(output_vec.begin(),output_vec.end()));
    //qDebug()<< "(TrtInfer::realTimeInfer)predicted label:"<<QString::number(pred);
    degrees=output_vec;
    *predIdx=pred;

}
//InferThread::run里调用
void TrtInfer::createEngine(std::string modelPath){
    if (readTrtFile(modelPath,modelStream, engine)) qDebug()<< "(TrtInfer::realTimeInfer)tensorRT engine created successfully." ;
    else qDebug()<< "(TrtInfer::realTimeInfer)tensorRT engine created failed." ;
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
    INFERENCE_BATCH=1;
    //define the dims if necessary
    if(isDynamic){
        nvinfer1::Dims dims4;   dims4.d[0]=INFERENCE_BATCH;
        for(int i=1;i<indims.nbDims;i++) dims4.d[i]=inputdims[i];
        dims4.nbDims = indims.nbDims;
        context->setBindingDimensions(0, dims4);
    }
}

void TrtInfer::setBatchSize(int batchSize){
    INFERENCE_BATCH=batchSize;
}

void TrtInfer::setParmsOfABFC(int modelIdxp, std::vector<int> dataOrderp){
    modelIdx=modelIdxp;
    dataOrder=dataOrderp;
}
