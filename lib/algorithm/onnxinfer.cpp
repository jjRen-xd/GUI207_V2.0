#include <torch/torch.h>
#include "onnxinfer.h"
#include <QDebug>
using namespace nvinfer1;
static Logger gLogger;

OnnxInfer::OnnxInfer(std::map<std::string, int> class2label):class2label(class2label){

}

bool read_TRT_File(const std::string& engineFile, IHostMemory*& trtModelStream, ICudaEngine*& engine)
{
    std::fstream file;
    std::cout << "loading filename from:" << engineFile << std::endl;
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
    std::cout << "load engine done" << std::endl;
    std::cout << "deserializing" << std::endl;
    trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    //ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
    engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    std::cout << "deserialize done" << std::endl;
    assert(engine != nullptr);
    std::cout << "Great. The engine in TensorRT.cpp is not nullptr" << std::endl;
    trtModelStream = engine->serialize();
    return true;
}

void getFloatFromTXT(std::string data_path,float* y) {
    int r, n = 0; double d; FILE* f;
    float temp[1024];
    f = fopen(data_path.c_str(), "r");
    for (int i = 0; i < 2; i++) fscanf(f, "%*[^\n]%*c"); // 跳两行
    for (int i = 0; i < 1024; i++) {
        r = fscanf(f, "%lf", &d);
        if (1 == r) temp[n++] = d;
        else if (0 == r) fscanf(f, "%*c");
        else break;
    }
    fclose(f);
    for (int i = 0; i < 512; i++) {
        y[i] = temp[i*2 + 1];
    }

    std::vector<float> features; //临时特征向量
    for (int d = 0; d < 512; ++d)
        features.push_back(y[d]);
    //特征归一化
    float dMaxValue = *std::max_element(features.begin(), features.end());  //求最大值
    float dMinValue = *std::min_element(features.begin(), features.end());  //求最小值
    for (int f = 0; f < features.size(); ++f) {
        y[f] = (y[f] - dMinValue) / (dMaxValue - dMinValue + 1e-8);
    }
    features.clear();//删除容器
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

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    void* buffers[2] = { NULL,NULL };
    cudaMalloc(&buffers[0], batchSize * 1 * 512 * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * 5 * sizeof(float));
    /*for (int i = 0; i < batchSize * 1 * 512; i++) {
        std::cout << input[i] << " ";
    }std::cout << std::endl<<"输出向量展示完毕"<<std::endl;*/

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    cudaMemcpyAsync(buffers[0], input, batchSize * 1 * 512 * sizeof(float), cudaMemcpyHostToDevice, stream);
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

void OnnxInfer::testOneSample(std::string targetPath, std::string modelPath, std::promise<int> *promisePredIdx, std::promise<std::vector<float>> *degree){
    if (read_TRT_File(modelPath,modelStream, engine)) std::cout << "tensorRT engine created successfully." << std::endl;
    else std::cout << "tensorRT engine created failed." << std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    float data[512] = { 0 };
    float prob[5] = { 0 };
    getFloatFromTXT(targetPath, data);
    doInference(*context, data, prob, 1);

    torch::Tensor output_tensor = torch::ones({1,5});
    std::cout << "output_tensor:  ";
    for (unsigned int i = 0; i < 5; i++){
        std::cout << prob[i] << ", ";
        output_tensor[0][i]=prob[i];
    }
    output_tensor = torch::softmax(output_tensor, 1).flatten();
    auto pred_tensor = output_tensor.argmax(0);
    int pred=pred_tensor.item<int>();
    //int classnum=5;
    std::cout <<std::endl<< "predicted label:"<<pred_tensor<<std::endl;
    std::vector<float> output(output_tensor.data_ptr<float>(),output_tensor.data_ptr<float>()+output_tensor.numel());
    degree->set_value(output);
    //emit finished(pred);
    promisePredIdx->set_value(pred);
    qDebug() << "subThread Done! pred=" << pred;
}

void OnnxInfer::testAllSample(std::string dataset_path,std::string model_path,float &Acc,std::vector<std::vector<int>> &confusion_matrix){
    if (read_TRT_File(model_path,modelStream, engine)) std::cout << "tensorRT engine created successfully." << std::endl;
    else std::cout << "tensorRT engine created failed." << std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);

    // LOAD DataSet
    auto test_dataset = dataSetClc(dataset_path, ".txt", class2label)
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();

    // batch : data_loader数据量为1
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 1);

    int correct=0,real=0,guess=0;
    float data_f[512] = { 0 };
    float prob_f[5] = { 0 };
    torch::Tensor output_tensor = torch::ones({1,5});
    for (const auto &batch : *test_loader){
        auto data_tensor = batch.data;
        auto targets_tensor = batch.target;
        targets_tensor.resize_({1});
        auto res = data_tensor.accessor<float,3>();            //拙劣待优化的tensor转float[]
        for(int i=0;i<512;i++){
            data_f[i]=res[0][0][i];
        }
        doInference(*context, data_f, prob_f, 1);
        for (unsigned int i = 0; i < 5; i++){
            std::cout << prob_f[i] << ", ";
            output_tensor[0][i]=prob_f[i];
        }
        auto pred = output_tensor.argmax(1);
        correct += pred.eq(targets_tensor).sum().template item<int64_t>();
        real=targets_tensor[0].item<int>();
        guess=output_tensor.view(-1).argmax().item<int>();
        confusion_matrix[real][guess]++;
    }

    std::cout << "correct:"<<correct<<std::endl;
    std::cout << "test_dataset_size:"<<test_dataset_size<<std::endl;
    Acc=static_cast<float> (correct) / (test_dataset_size);
}
