#include "matdataprocess_abfc.h"

MatDataProcess_abfc::MatDataProcess_abfc(std::vector<int> dataOrder,int modelIdx):
    dataOrder(dataOrder),modelIdx(modelIdx){
    
}
//TODO 重名可能有问题
void MatDataProcess_abfc::oneNormalization(std::vector<float> &list){
    //特征归一化
    float dMaxValue = *max_element(list.begin(),list.end());  //求最大值
    //std::cout<<"maxdata"<<dMaxValue<<'\n';
    float dMinValue = *min_element(list.begin(),list.end());  //求最小值
    //std::cout<<"mindata"<<dMinValue<<'\n';
    for (int f = 0; f < list.size(); ++f) {
        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//极小值限制
    }
}

void MatDataProcess_abfc::getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen,std::vector<int> &eachClassQuantity){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_abfc:getAllDataFromMat)文件指针空！！！！！！";
        return;
    }
    std::string matVariable=matPath.substr(
                matPath.find_last_of('/')+1,
                matPath.find_last_of('.')-matPath.find_last_of('/')-1).c_str();////假设数据变量名同文件名的话
    //qDebug()<<"(MatDataProcess_abfc:getAllDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(MatDataProcess_abfc:getAllDataFromMat)pMxArray变量没找到！！！！！！";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    //考虑用户给的数据集M不等于模型对应的dataOrder.size()
    if(!(dataOrder.size()==M)){
        qDebug()<<"(MatDataProcess_abfc::getAllDataFromMat) 所选数据集和模型或其权重向量长度不匹配！";
        return ;
    }
    int thisClassNum=0;
    //for(int i=0;i<N/2;i++){
    for(int i=N/2;i<N;i++){
        std::vector<float> onesmp;//存当前遍历的一个样本
        for(int j=0;j<M;j++){
            onesmp.push_back(matdata[i*M+j]);
        }
        
        std::vector<float> onesmp_fixed(onesmp.size());
        for(int j=0;j<onesmp.size();j++) onesmp_fixed[j]=onesmp[dataOrder[j]];//fix order
        if(dataProcess) oneNormalization(onesmp_fixed);//归一化
        
        // if(matVariable=="Cone"&&i<51){
        //     std::cout<<std::endl<<"(MatDataProcess_abfc::getAllDataFromMat)Cone.mat the "<<i<<"col data:"<<std::endl;
        //     for(int p=0;p<onesmp.size();p++) std::cout<<onesmp[p]<<" ";
        // }

        //erase data
        std::vector<float> temp;
        if(modelIdx>M){
            modelIdx=M;
            qDebug()<<"(MatDataProcess_abfc::getAllDataFromMat) 欲选择的特征数量大于实际最大,自适应选取所有";
        }
        for(int j=0;j<modelIdx;j++){
            temp.push_back(onesmp_fixed[j]);
        }
        //std::cout<<&temp<<std::endl;

        data.push_back(temp);
        labels.push_back(label);
        thisClassNum++;
    }
    // qDebug()<<"(MatDataProcess_abfc:getAllDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    eachClassQuantity.push_back(thisClassNum);
}

void MatDataProcess_abfc::loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,std::map<std::string, int> &class2label,int inputLen,std::vector<int> &eachClassQuantity){
    SearchFolder *dirTools = new SearchFolder();
    // 寻找子文件夹 WARN:数据集的路径一定不能包含汉字 否则遍历不到文件路径
    std::vector<std::string> subDirs;
    dirTools->getDirs(subDirs, datasetPath);
    for(auto &subDir: subDirs){
        // 寻找每个子文件夹下的样本文件
        std::vector<std::string> fileNames;
        std::string subDirPath = datasetPath+"/"+subDir;
        dirTools->getFiles(fileNames, type, subDirPath);
        for(auto &fileName: fileNames){//一般就一个mat
            qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName)<<" label:"<<class2label[subDir];
            getAllDataFromMat(subDirPath+"/"+fileName,false,data,labels,class2label[subDir],inputLen,eachClassQuantity);
        }
    }
    if(dataProcess){
        for(int i=0;i<data[0].size();i++){//遍历n个特征个次数，对数据集的每个特征做归一
            std::vector<float> datasetMatrix_row;
            for(int j=0;j<data.size();j++){
                datasetMatrix_row.push_back(data[j][i]);
            }
            oneNormalization(datasetMatrix_row);//归一化
            for(int j=0;j<data.size();j++){
                data[j][i]=datasetMatrix_row[j];
            }
        }
    }
    qDebug()<<"(MatDataProcess_abfc::loadAllDataFromFolder)data.size()="<<data.size();
    qDebug()<<"(MatDataProcess_abfc::loadAllDataFromFolder)data[0].size()="<<data[0].size();
    return;
}

void MatDataProcess_abfc::getDataFromMat(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(targetMatFile.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_abfc:getDataFromMat)文件指针空！！！！！！";
        return;
    }

    std::string matVariable=targetMatFile.substr(
                targetMatFile.find_last_of('/')+1,
                targetMatFile.find_last_of('.')-targetMatFile.find_last_of('/')-1).c_str();//假设数据变量名同文件名的话
    qDebug()<<"(MatDataProcess_abfc:getDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(MatDataProcess_abfc:getDataFromMat)pMxArray变量没找到！！！！！！";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    //if(!(dataOrder.size()==M && inputLen==M)){
    if(!(dataOrder.size()==M)){
        qDebug()<<"(MatDataProcess_abfc::getAllDataFromMat) 所选数据集和模型或其权重向量长度不匹配！";
        return ;
    }
    if(emIdx>N) emIdx=N-1;  //说明是随机数

    std::vector<float> onesmp;//存当前样本
    for(int i=0;i<M;i++){
        onesmp.push_back(matdata[emIdx*M+i]);
    }
    std::vector<float> onesmp_fixed(onesmp.size());
    for(int j=0;j<onesmp.size();j++) onesmp_fixed[j]=onesmp[dataOrder[j]];//fix order
    if(dataProcess) oneNormalization(onesmp_fixed);//归一化

    for(int i=0;i<inputLen;i++){
        data[i]=onesmp_fixed[i];
    }

}
