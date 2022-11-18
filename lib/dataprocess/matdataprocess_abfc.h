#ifndef MATDATAPROCESS_ABFC_H
#define MATDATAPROCESS_ABFC_H

#include <mat.h>
#include <vector>
#include <map>
#include <QDebug>
#include "./lib/guiLogic/tools/searchFolder.h"

class MatDataProcess_abfc
{
public:
    MatDataProcess_abfc(std::vector<int> dataOrder,int modelIdx);
    ~MatDataProcess_abfc(){};
    void oneNormalization(std::vector<float> &list);

    void getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen,std::vector<int> &eachClassQuantity);
    void loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,
                                std::vector<std::vector<float>> &data,std::vector<int> &labels,
                                std::map<std::string, int> &class2label,int inputLen,std::vector<int> &eachClassQuantity);
    void getDataFromMat(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen);
    
private:
    std::vector<int> dataOrder;
    int modelIdx;
};

#endif // MATDATAPROCESS_ABFC_H
