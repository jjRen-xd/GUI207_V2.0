#include "libtorchTest.h"

LibtorchTest::LibtorchTest(std::map<std::string, int> class2label):class2label(class2label){

}

LibtorchTest::~LibtorchTest(){

}


torch::Tensor getTensorFromTXT(std::string data_path){
    int r, n = 0;
    double d;
    FILE *f;
    float temp[1024];
    f = fopen(data_path.c_str(), "r");
//    for (int i = 0; i < 2; i++)
//        fscanf(f, "%*[^\n]%*c"); // 跳两行
    for (int i = 0; i < 1024; i++){
        r = fscanf(f, "%lf", &d);
    if (1 == r)
        temp[n++] = d;
    else if (0 == r)
        fscanf(f, "%*c");
    else
        break;
    }
    fclose(f);
    float x[512], y[512];
    for (int i = 0; i < 512; i++){
//        x[i] = temp[i];
        y[i] = temp[2*i + 1];
    }
//    torch::Tensor t1 = torch::from_blob(x, {512}, torch::kFloat);
    torch::Tensor t2 = torch::from_blob(y, {512}, torch::kFloat);
//    t1 = (t1 - t1.min()) / (t1.max() - t1.min());
    t2 = (t2 - t2.min()) / (t2.max() - t2.min());
    torch::Tensor t = torch::cat({t2}, 0).view({1, 512});

    return t.clone();
}


struct Net: torch::nn::Module
{
  Net()
      : conv1(torch::nn::Conv1dOptions(1, 16, /*kernel_size=*/5)),
        conv2(torch::nn::Conv1dOptions(16, 32, /*kernel_size=*/5)),
        fc1(4000, 512),
        fc2(512, 5)
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    // register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(torch::max_pool1d(conv1->forward(x), 2));
    x = torch::relu(torch::max_pool1d(conv2->forward(x), 2));
    x = x.view({-1, 4000});
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv1d conv1;
  torch::nn::Conv1d conv2;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};


int LibtorchTest::testOneSample(std::string targetPath, std::string modelPath, std::vector<float> &degree, double &predTime){
    // model;
    torch::Device device(torch::kCPU);
    Net model;
    model.to(device);
    torch::serialize::InputArchive archive;
    archive.load_from(modelPath);
    model.load(archive);
    model.eval();
    std::cout << "Model Load Successfully"<<std::endl;

    clock_t start_time, end_time;
    start_time = clock();

    // dataset
    torch::Tensor data_tensor = getTensorFromTXT(targetPath);   //[CPUFloatType [2, 512]]
    auto data = data_tensor.to(device);
    data = data.to(torch::kFloat32).unsqueeze(0);
//    data.print();
    // forward
    auto output_tensor = model.forward(data);
    output_tensor = torch::softmax(output_tensor, 1).flatten();
    auto pred_tensor = output_tensor.argmax(0);

    end_time = clock();
    predTime = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    std::cout << "Predict Time: " << predTime << std::endl;

//    std::cout<<output_tensor<<std::endl;
//    std::cout<<pred_tensor<<std::endl;

    std::vector<float> output(output_tensor.data_ptr<float>(),output_tensor.data_ptr<float>()+output_tensor.numel());
    degree = output;

    int pred = pred_tensor.item<int>();
    return pred;
}



/*待优化*/
void load_data_from_folder(
    std::string datasetPath,
    std::string type,
    std::vector<std::string> &dataPaths,
    std::vector<int> &labels,
    std::map<std::string, int> &class2label
){
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
            dataPaths.push_back(subDirPath+"/"+fileName);
            labels.push_back(class2label[subDir]);
        }
    }
    return;
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


void LibtorchTest::testAllSample(std::string dataset_path,std::string model_path,float &Acc,std::vector<std::vector<int>> &confusion_matrix){
    std::cout<<dataset_path<<" "<<model_path<<std::endl;
    torch::Device device(torch::kCPU);
    Net model;
    model.to(device);
    // LOAD Model
    torch::serialize::InputArchive archive;
    archive.load_from(model_path);
    model.load(archive);
    model.eval();
    std::cout << "Model Load Successfully" <<std::endl;

    // LOAD DataSet
    auto test_dataset = dataSetClc(dataset_path, ".txt", class2label)
            .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();

    // batch : data_loader数据量为1
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 1);

    int correct=0,real=0,guess=0;
    for (const auto &batch : *test_loader){
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        targets.resize_({1});
        targets = targets.to(torch::kInt64);
        data = data.to(torch::kFloat32);
        auto output = model.forward(data);
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
        real=targets[0].item<int>();
        guess=output.view(-1).argmax().item<int>();
        confusion_matrix[real][guess]++;
    }

    std::cout << "correct:"<<correct<<std::endl;
    std::cout << "test_dataset_size:"<<test_dataset_size<<std::endl;
    Acc=static_cast<float> (correct) / (test_dataset_size);
}


