#ifndef MATDATAPROCESS_AFS_H
#define MATDATAPROCESS_AFS_H

#include <mat.h>
#include <vector>
#include <map>
#include <QDebug>
#include "./lib/guiLogic/tools/searchFolder.h"

class MatDataProcess_afs
{
public:
    MatDataProcess_afs(std::vector<int> dataOrder,int modelIdx);
    ~MatDataProcess_afs(){};
    void oneNormalization(std::vector<float> &list);

    void getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen);
    void loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,
                                std::vector<std::vector<float>> &data,std::vector<int> &labels,
                                std::map<std::string, int> &class2label,int inputLen);
    void getDataFromMat(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen);
    
private:
    std::vector<int> dataOrder;
    int modelIdx;
};

#endif // MATDATAPROCESS_AFS_H
