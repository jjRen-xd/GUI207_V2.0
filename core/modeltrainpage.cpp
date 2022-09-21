#include "modelTrainPage.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include <QMessageBox>
#include <QFileDialog>
#include <windows.h>
#include <mat.h>

ModelTrainPage::ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo):
    ui(main_ui),terminal(bash_terminal),datasetInfo(globalDatasetInfo),modelInfo(globalModelInfo){

    processTrain = new ModelTrain(ui->textBrowser, ui->train_img, ui->val_img, ui->confusion_mat, ui->trainProgressBar);

//    connect(ui->datadirButton, &QPushButton::clicked, this, &ModelTrainPage::chooseDataDir);
//    connect(ui->starttrianButton, &QPushButton::clicked, this, &ModelTrainPage::startTrain);
//    connect(ui->stoptrainButton, &QPushButton::clicked, this, &ModelTrainPage::stopTrain);
    connect(ui->editModelButton, &QPushButton::clicked, this, &ModelTrainPage::editModelFile);
    connect(ui->modeltypeBox, &QComboBox::currentIndexChanged, this, &ModelTrainPage::changeTrainType);
//    connect(ui->oldClassBox, &QComboBox::activated, this, &ModelTrainPage::chooseOldClass);


    ui->stackedWidget->setCurrentIndex(0);
}
/*
void ModelTrainPage::chooseDataDir(){
    QString dataPath = QFileDialog::getExistingDirectory(NULL,"请选择待训练数据的根目录","./",QFileDialog::ShowDirsOnly);
    if(dataPath == ""){
        QMessageBox::warning(NULL,"提示","未选择有效数据集根目录!");
        ui->datadirEdit->setText("");
        return;
    }
    ui->datadirEdit->setText(dataPath);

    int len=getDataClassNum(dataPath.toStdString(), "model_saving");
    ui->oldClassBox->clear();
    for(int i=1;i<len;i++){
        ui->oldClassBox->addItem(QString::number(i));
    }
}

void ModelTrainPage::changeTrainType(){
    int modelType=ui->modeltypeBox->currentIndex();

    ui->tabWidget->removeTab(0);
    ui->tabWidget->removeTab(1);
    ui->tabWidget->removeTab(2);
    if(modelType==0){
        ui->tabWidget->addTab(ui->tab,"训练集准确率");
        ui->tabWidget->addTab(ui->tab_2,"验证集准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
        ui->stackedWidget->setCurrentIndex(0);
    }
    else if(modelType==1){
        ui->tabWidget->addTab(ui->tab_2,"特征准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
        ui->stackedWidget->setCurrentIndex(0);
    }
    else if(modelType==2){
        ui->tabWidget->addTab(ui->tab_2,"特征准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
        ui->stackedWidget->setCurrentIndex(1);
    }
}

void ModelTrainPage::startTrain(){
    int modelType=ui->modeltypeBox->currentIndex();
    QString dataDir = ui->datadirEdit->toPlainText();
    if(dataDir==""){
        QMessageBox::warning(NULL, "配置出错", "请指定待训练数据的根目录!");
        return;
    }
    QString cmd;
    if(modelType<2){
        //TODO 此处判断模型类型和数据是否匹配
        QString bathSize=ui->batchsizeBox->currentText();
        QString maxEpoch=ui->maxepochBox->currentText();
        switch(modelType){
            case 0:cmd = "activate TF2 && python ../../db/bashs/hrrp/train.py"
                        " --data_dir "+dataDir+" --batch_size "+bathSize+" --max_epochs "+maxEpoch;break;
            case 1:cmd = "activate TF2 && python ../../db/bashs/afs/train.py"
                        " --data_dir "+dataDir+" --batch_size "+bathSize+" --max_epochs "+maxEpoch;break;
        }
    }
    else if(modelType==2){
        //TODO 此处判断模型类型和数据是否匹配
        int allclassNum=getDataClassNum(dataDir.toStdString(), "model_saving");
        int data_dimension=getDataLen(dataDir.toStdString());
        QString oldclassNum=ui->oldClassBox->currentText();
        QString sampleRatio=ui->sampleRatioBox->currentText();
        QString bathSize=ui->batchSizeBox2->currentText();
        QString preEpoch=ui->pretrainEpochBox->currentText();
        QString addEpoch=ui->increseEpochBox->currentText();

        cmd="activate PT && python ../../db/bashs/incremental/main.py --all_class="+QString::number(allclassNum)+
                " --batch_size="+bathSize+" --bound="+sampleRatio+" --increment_epoch="+addEpoch+
                " --learning_rate=0.001 --memory_size=200 --old_class="+oldclassNum+" --pretrain_epoch="+
                preEpoch+" --random_seed=2022 --snr=2 --task_size=1 --test_ratio=0.5 --data_dimension="+
                QString::number(data_dimension)+" --raw_data_path="+dataDir;
    }
    qDebug() << cmd;
    processTrain->startTrain(modelType, cmd);
}

void ModelTrainPage::stopTrain(){
    processTrain->stopTrain();
}

void ModelTrainPage::chooseOldClass(){
    std::string dataPath;
    if(ui->datadirEdit->toPlainText().toStdString()==""){
        QMessageBox::warning(NULL, "操作提醒", "请先指定待训练数据的根目录在选择该参数!");
        return;
    }
}

void ModelTrainPage::editModelFile(){
    int modelType=ui->modeltypeBox->currentIndex();
    QString modelFilePath;
    switch(modelType){
        case 0:modelFilePath="../../db/bashs/hrrp/train.py";break;
        case 1:modelFilePath="../../db/bashs/afs/afs_model.py";break;
        case 2:modelFilePath="../../db/bashs/incremental/model.py";break;
    }
    QString commd="gvim " + modelFilePath;
    WinExec(commd.toStdString().c_str(), SW_HIDE);
}

int ModelTrainPage::getDataClassNum(std::string dataPath, std::string specialDir){
    std::vector<std::string> allDir;
    SearchFolder searchFolder;
    if (searchFolder.getDirs(allDir, dataPath)) {
        auto tar = std::find(allDir.begin(), allDir.end(), specialDir);
        if (tar != allDir.end()) return allDir.size() - 1;
        else return allDir.size();
    }
}

int ModelTrainPage::getDataLen(std::string dataPath){//指定数据集地址，返回其中mat数据的长度  #include<mat.h>
    std::string theMatFilePath;
    std::string matVariable;
    std::vector<std::string> allMatFile;
    std::vector<std::string> allDir;
    SearchFolder searchFolder;
    int dataLen=-1;
    if(searchFolder.getDirs(allDir, dataPath)){
        if(searchFolder.getFiles(allMatFile, ".mat", dataPath+"/"+allDir[0])){
            theMatFilePath=dataPath+"/"+allDir[0]+"/"+allMatFile[0];
            matVariable=allMatFile[0].substr(0,allMatFile[0].find_last_of('.')).c_str();//假设数据变量名同文件名的话
        }
    }
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    pMatFile = matOpen(theMatFilePath.c_str(), "r");
    if(!pMatFile){qDebug()<<"()文件指针空！！！！！！";return -1;}
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){qDebug()<<"()pMxArray变量没找到！！！！！！";return -1;}
    dataLen = mxGetM(pMxArray);  //N 列数
    return dataLen;
}*/
