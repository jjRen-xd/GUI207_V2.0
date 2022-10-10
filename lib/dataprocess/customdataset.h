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
    std::vector<int> eachClassQuantity;

    CustomDataset(std::string dataSetPath, bool dataProcess, std::string type, std::map<std::string, int> class2label,int inputLen,std::string flag="TRA_DL",int modelIdx=1,std::vector<int> dataOrder=std::vector<int>())
        :class2label(class2label){
        if(flag=="FEA_RELE"){
            MatDataProcess_afs matDataPrcs(dataOrder,modelIdx);
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type,dataProcess, data, labels, class2label,inputLen,eachClassQuantity);
        }
        else{
            MatDataProcess matDataPrcs;
            matDataPrcs.loadAllDataFromFolder(dataSetPath, type,dataProcess, data, labels, class2label,inputLen);
        }
        
    }
    int size(){
        return labels.size();
    };
    void getDataSpecifically(std::string theClass,int emIndex,float *dataf){
        int theClassIdx=class2label[theClass];
        std::map<std::string, int>::iterator iter;
        iter = class2label.begin();
        while(iter != class2label.end()) {
            qDebug() << QString::fromStdString(iter->first) << " : " << iter->second ;
            iter++;
        }
        int globalIndexOnValidationSet = 0;
        for(int i=0;i<theClassIdx;i++){
            globalIndexOnValidationSet += eachClassQuantity[i];
        }
        //TODO 要灵活 之后根据比例来
        if(emIndex>=50)emIndex-=50;
        globalIndexOnValidationSet += emIndex;
        for(int i=0;i<data[globalIndexOnValidationSet].size();i++){
            dataf[i]=data[globalIndexOnValidationSet][i];
        }
    }
};
