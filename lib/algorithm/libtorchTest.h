#ifndef LIBTORCHTEST_H
#define LIBTORCHTEST_H

#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS

#include <QObject>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include <time.h>

#include "./lib/guiLogic/tools/searchFolder.h"

//#include "./lib/guiLogic/bashTerminal.h"

torch::Tensor getTensorFromTXT(std::string data_path);
void load_data_from_folder(
    std::string datasetPath,
    std::string type,
    std::vector<std::string> &dataPaths,
    std::vector<int> &labels,
    std::map<std::string, int> &class2label
);

class LibtorchTest: public QObject{
    Q_OBJECT

    public:
        LibtorchTest(std::map<std::string, int> class2label);
        ~LibtorchTest();

    public slots:
        int testOneSample(std::string targetPath, std::string modelPath, std::vector<float> &degree, double &predTime);
        void testAllSample(std::string hrrpdataset_path,std::string hrrpmodel_path,float &Acc,std::vector<std::vector<int>> &confusion_matrix);
    private:
//        BashTerminal *terminal;
        // 不同平台下文件夹搜索工具
        std::map<std::string, int> class2label;
};

#endif // LIBTORCHTEST_H
