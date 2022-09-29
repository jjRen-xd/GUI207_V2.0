#pragma once
#include "./lib/dataprocess/matdataprocess.h"
#include "./lib/dataprocess/matdataprocess_afs.h"
#include <vector>
#include <map>

//CustomDataSet.data=F(The M of dataset's mat , dims(model.inputlayer))
class CustomDataset{
public:
    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    std::map<std::string, int> class2label;
    CustomDataset(std::string dataSetPath, bool dataProcess, std::string type, std::map<std::string, int> class2label,int inputLen,std::string flag="TRA_DL",int modelIdx=1,std::vector<int> dataOrder=std::vector<int>())
        :class2label(class2label){
        if(flag=="FEA_RELE"){
            MatDataProcess_afs matDataPrcs(dataOrder,modelIdx);
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type,dataProcess, data, labels, class2label,inputLen);
        }
        else{
            MatDataProcess matDataPrcs;
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type,dataProcess, data, labels, class2label,inputLen);
        }
        
    }
    int size(){
        return labels.size();
    };
};
