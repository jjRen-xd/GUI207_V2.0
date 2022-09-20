#include "matdataprocess.h"

void MatDataProcess::oneNormalization(std::vector<float> &list){
    //������һ��
    float dMaxValue = *max_element(list.begin(),list.end());  //�����ֵ
    //std::cout<<"maxdata"<<dMaxValue<<'\n';
    float dMinValue = *min_element(list.begin(),list.end());  //����Сֵ
    //std::cout<<"mindata"<<dMinValue<<'\n';
    for (int f = 0; f < list.size(); ++f) {
        list[f] = (1-0)*(list[f]-dMinValue)/(dMaxValue-dMinValue+1e-8)+0;//��Сֵ����
    }
}

void MatDataProcess::getAllDataFromMat(std::string matPath,bool dataProcess,std::vector<std::vector<float>> &data,std::vector<int> &labels,int label,int inputLen){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // ��ȡ.mat�ļ�������mat�ļ���Ϊ"initUrban.mat"�����а���"initA"��
    double* matdata;
    pMatFile = matOpen(matPath.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess:getAllDataFromMat)�ļ�ָ��գ�����������";
        return;
    }
    std::string matVariable=matPath.substr(
                matPath.find_last_of('/')+1,
                matPath.find_last_of('.')-matPath.find_last_of('/')-1).c_str();//�������ݱ�����ͬ�ļ����Ļ�
    //qDebug()<<"(MatDataProcess:getAllDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getAllDataFromMat)pMxArray����û�ҵ�������������";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //����
    int N = mxGetN(pMxArray);  //����

    for(int i=0;i<N;i++){
        std::vector<float> onesmp;//�浱ǰ������һ������
        for(int j=0;j<M;j++){
            onesmp.push_back(matdata[i*M+j]);
        }
        if(dataProcess) oneNormalization(onesmp);//��һ��
        std::vector<float> temp;
        for(int j=0;j<inputLen;j++){
            temp.push_back(onesmp[j%M]);//���inputLen��N��С�����ᱨ������Ȼ���ݼ���ģ���ǲ���Ӧ�İɣ��õ���������Ӧ����ѿ�
        }
        //std::cout<<&temp<<std::endl;
        data.push_back(temp);
        labels.push_back(label);
    }
}

void MatDataProcess::loadAllDataFromFolder(std::string datasetPath,std::string type,bool dataProcess,std::vector<std::vector<float>> &data,
                           std::vector<int> &labels,std::map<std::string, int> &class2label,int inputLen){
    SearchFolder *dirTools = new SearchFolder();
    // Ѱ�����ļ��� WARN:���ݼ���·��һ�����ܰ������� ������������ļ�·��
    std::vector<std::string> subDirs;
    dirTools->getDirs(subDirs, datasetPath);
    for(auto &subDir: subDirs){
        // Ѱ��ÿ�����ļ����µ������ļ�
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
    // ��ȡ.mat�ļ�������mat�ļ���Ϊ"initUrban.mat"�����а���"initA"��
    double* matdata;
    pMatFile = matOpen(targetMatFile.c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess:getDataFromMat)�ļ�ָ��գ�����������";
        return;
    }

    std::string matVariable=targetMatFile.substr(
                targetMatFile.find_last_of('/')+1,
                targetMatFile.find_last_of('.')-targetMatFile.find_last_of('/')-1).c_str();//�������ݱ�����ͬ�ļ����Ļ�
    qDebug()<<"(MatDataProcess:getDataFromMat)matVariable=="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getDataFromMat)pMxArray����û�ҵ�������������";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //M����
    int N = mxGetN(pMxArray);  //N ����
    if(emIdx>N) emIdx=N-1; //˵���������

    std::vector<float> onesmp;//�浱ǰ����
    for(int i=0;i<M;i++){
        onesmp.push_back(matdata[emIdx*M+i]);
    }
    if(dataProcess) oneNormalization(onesmp);//��һ��
    for(int i=0;i<inputLen;i++){
        data[i]=onesmp[i%M];//matlab���д洢
    }

}