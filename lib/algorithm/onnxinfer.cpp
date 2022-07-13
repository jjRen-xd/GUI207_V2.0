#include <torch/torch.h>
#include "onnxinfer.h"
#include <climits>

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
    std::cout << "length:" << length << std::endl;
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
    std::cout << "The engine in TensorRT.cpp is not nullptr" << std::endl;
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
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    //const ICudaEngine& engine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    //assert(engine.getNbBindings() == 2);
    void* buffers[2] = { NULL,NULL };
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    //const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    //const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    cudaMalloc(&buffers[0], batchSize * 1 * 512 * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * 5 * sizeof(float));


    /*for (int i = 0; i < batchSize * INPUT_H * INPUT_W; i++) {
        std::cout << input[i] << " ";
    }std::cout << std::endl<<"输出向量展示完毕"<<std::endl;*/

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    cudaMemcpyAsync(buffers[0], input, batchSize * 1 * 512 * sizeof(float), cudaMemcpyHostToDevice, stream);
    //开始推理
    std::cout << "start to infer ..." << std::endl;
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], batchSize * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    std::cout << "Inference Done." << std::endl;
}
int OnnxInfer::testOneSample(std::string targetPath, std::string modelPath, std::vector<float> &degree){
    if (read_TRT_File(modelPath,modelStream, engine)) std::cout << "tensorRT engine created successfully." << std::endl;
    else std::cout << "tensorRT engine created failed." << std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    float data[512] = { 0 };
    float prob[5] = { 0 };
    getFloatFromTXT(targetPath, data);
    doInference(*context, data, prob, 1);

    torch::Tensor output_tensor = torch::ones({1,5});
    std::cout << "Output:  ";
    for (unsigned int i = 0; i < 5; i++){
        std::cout << prob[i] << ", ";
        output_tensor[0][i]=prob[i];
    }
    output_tensor = torch::softmax(output_tensor, 1).flatten();
    auto pred_tensor = output_tensor.argmax(0);
    std::cout<<output_tensor<<std::endl;
    std::cout<<pred_tensor<<std::endl;
    int pred=pred_tensor.item<int>();

    std::vector<float> output(output_tensor.data_ptr<float>(),output_tensor.data_ptr<float>()+output_tensor.numel());
    degree = output;
    return pred;
}
