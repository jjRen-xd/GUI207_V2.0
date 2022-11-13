#include "matdataprocess.h"

void MatDataProcess::oneNormalization(std::vector<float> &list){
    //特征归一化
    float dMaxValue = *max_element(list.begin(),list.end());  //求最大值
    //std::cout<<"maxdata"<<dMaxValue<<'\n';
    float dMinValue = *min_element(list.begin(),list.end());  //求最小值
    //std::cout<<"mindata"<<dMinValue<<'\n';
    for (int f = 0; f < list.size(); ++f) {
        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//极小值限制
    }
}

void MatDataProcess::getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess:getAllDataFromMat)文件指针空！！！！！！";
        return;
    }
    std::string matVariable=matPath.substr(
                matPath.find_last_of('/')+1,
                matPath.find_last_of('.')-matPath.find_last_of('/')-1).c_str();////假设数据变量名同文件名的话
    //qDebug()<<"(MatDataProcess:getAllDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getAllDataFromMat)pMxArray变量没找到！！！！！！";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数

    //for(int i=0;i<N/2;i++){
    for(int i=N/2;i<N;i++){
        std::vector<float> onesmp;//存当前遍历的一个样本
        for(int j=0;j<M;j++){
            onesmp.push_back(matdata[i*M+j]);
        }
        if(dataProcess) oneNormalization(onesmp);//归一化
        std::vector<float> temp;
        int numberOfcopies=inputLen/M; //复制次数=网络的输入长度/一个样本数据的长度
        for(int j=0;j<inputLen;j++){
            //如果inputLen比N还小，不会报错，但显然数据集和模型是不对应的吧，得到的推理结果应会很难看
            temp.push_back(onesmp[j/numberOfcopies]);//64*128,对应训练时(128,64,1)的输入维度
            //temp.push_back(onesmp[j%M]);//128*64,对应训练时(64,128,1)的输入维度
        }

        data.push_back(temp);
        labels.push_back(label);
    }
    // qDebug()<<"(MatDataProcess:getAllDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
}

void MatDataProcess::loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,std::vector<std::vector<float>> &data,
                           std::vector<int> &labels,std::map<std::string, int> &class2label,int inputLen){
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
            //qDebug()<<QString::fromStdString(subDirPath)<<"/"<<QString::fromStdString(fileName)<<" label:"<<class2label[subDir];
            getAllDataFromMat(subDirPath+"/"+fileName,dataProcess,data,labels,class2label[subDir],inputLen);
        }
    }
    return;
}

void MatDataProcess::getDataFromMat(std::string targetMatFile,int emIdx,bool dataProcess,float *data,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(targetMatFile.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess:getDataFromMat)文件指针空！！！！！！";
        return;
    }

    std::string matVariable=targetMatFile.substr(
                targetMatFile.find_last_of('/')+1,
                targetMatFile.find_last_of('.')-targetMatFile.find_last_of('/')-1).c_str();//假设数据变量名同文件名的话
    qDebug()<<"(MatDataProcess:getDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getDataFromMat)pMxArray变量没找到！！！！！！";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    if(emIdx>N) emIdx=N-1;  //说明是随机数

    std::vector<float> onesmp;//存当前样本
    for(int i=0;i<M;i++){
        onesmp.push_back(matdata[emIdx*M+i]);
    }
    if(dataProcess) oneNormalization(onesmp);
    int numberOfcopies=inputLen/M; //复制次数=网络的输入长度/一个样本数据的长度
    for(int i=0;i<inputLen;i++){
        //data[i]=onesmp[i%M];//matlab按列存储
        data[i]=onesmp[i/numberOfcopies];//网络如果是(128,64,1),应该64+64+64+64+...输入引擎,而不是128+128+...
    }

}