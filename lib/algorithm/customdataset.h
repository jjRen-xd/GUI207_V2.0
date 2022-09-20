#pragma once
#include "./lib/algorithm/matdataprocess.h"
#include <vector>
#include <map>

class CustomDataset{
public:
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    std::map<std::string, int> class2label;
    CustomDataset(std::string dataSetPath, bool dataProcess, std::string type, std::map<std::string, int> class2label,int inputLen)
        :class2label(class2label){
        MatDataProcess matDataPrcs;
        matDataPrcs.loadAllDataFromFolder(dataSetPath, type,dataProcess, data, labels, class2label,inputLen);
    }
    int size(){
        return labels.size();
    };
};