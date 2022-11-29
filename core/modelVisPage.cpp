#include "modelVisPage.h"

#include "./lib/guiLogic/tinyXml/tinyxml.h"
#include "./lib/guiLogic/tools/convertTools.h"
#include "./core/datasetsWindow/chart.h"
#include <QGraphicsScene>
#include <QMessageBox>
#include <QFileDialog>
#include <mat.h>

using namespace std;
#define NUM_FEA 6

ModelVisPage::ModelVisPage(Ui_MainWindow *main_ui,
                             BashTerminal *bash_terminal,
                             DatasetInfo *globalDatasetInfo,
                             ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo)
{   
    // 刷新模型、数据集信息
//    this->condaPath = "/home/z840/anaconda3/bin/activate";
    this->condaEnvName = "torch";
    this->pythonApiPath = "../../api/HRRP_vis/vis_fea.py";
    refreshGlobalInfo();

    // 下拉框信号槽绑定
    connect(ui->comboBox_mV_L1, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L1(QString)));
    connect(ui->comboBox_mV_L2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L2(QString)));
    connect(ui->comboBox_mV_L3, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L3(QString)));
    connect(ui->comboBox_mV_L4, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L4(QString)));
    connect(ui->comboBox_mV_L5, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L5(QString)));

    // 按钮信号槽绑定
    connect(ui->pushButton_mV_clear, &QPushButton::clicked, this, &ModelVisPage::clearComboBox);
    connect(ui->pushButton_mV_randomImg, &QPushButton::clicked, this, &ModelVisPage::randomImage);
    connect(ui->pushButton_mV_confirmVis, &QPushButton::clicked, this, &ModelVisPage::confirmVis);
    connect(ui->pushButton_mV_nextPage, &QPushButton::clicked, this, &ModelVisPage::nextFeaImgsPage);
    // 多线程的信号槽绑定
    processVis = new QProcess();
    connect(processVis, &QProcess::readyReadStandardOutput, this, &ModelVisPage::processVisFinished);

}

ModelVisPage::~ModelVisPage(){

}


void ModelVisPage::confirmVis(){
    // 各种参数的校验
    if(this->choicedSamplePATH.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择输入图像!");
        return;
    }
    if(this->modelCheckpointPath.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选中模型或不支持该类型模型!");
        return;
    }
    if(this->targetVisLayer.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择可视化隐层!");
        return;
    }
    if(ui->comboBox_mV_actOrGrad->currentText()=="神经元激活"){
        this->actOrGrad = 0;
    }
    else if(ui->comboBox_mV_actOrGrad->currentText()=="神经元梯度"){
        this->actOrGrad = 1;
    }
    else{
        QMessageBox::warning(NULL,"错误","需选择神经元激活/梯度!");
        return;
    }

    // 激活conda python环境
    if (this->choicedModelSuffix == ".pth"){        // pytorch模型
        this->condaEnvName = "torch";
        this->pythonApiPath = "../../api/HRRP_vis/hrrp_vis_torch.py";
    }
    else if(this->choicedModelSuffix == ".hdf5"){   // keras模型
        this->condaEnvName = "keras";
        this->pythonApiPath = "../../api/HRRP_vis_keras/hrrp_vis_keras.py";
    }
    else{
        QMessageBox::warning(NULL,"错误","不支持该类型模型!");
        return;
    }

    // 执行python脚本
    QString activateEnv = "conda activate "+this->condaEnvName+"&&";
    QString command = activateEnv + "python " + this->pythonApiPath+ \
        " --checkpoint="        +this->modelCheckpointPath+ \
        " --visualize_layer="   +this->targetVisLayer+ \
        " --mat_path="          +this->choicedSamplePATH+ \
        " --mat_idx="           +QString::number(this->choicedMatIdx)+ \
        " --act_or_grad="       +QString::number(this->actOrGrad)+ \
        " --save_path="         +this->feaImgsSavePath;
    this->terminal->print(command);
    this->execuCmdProcess(command);
}


// 可视化线程
void ModelVisPage::execuCmdProcess(QString cmd){
    if(processVis->state()==QProcess::Running){
        processVis->close();
        processVis->kill();
    }
    processVis->setProcessChannelMode(QProcess::MergedChannels);
    processVis->start("cmd.exe");
    ui->progressBar_mV_visFea->setMaximum(0);
    ui->progressBar_mV_visFea->setValue(0);
    processVis->write(cmd.toLocal8Bit() + '\n');
}


void ModelVisPage::processVisFinished(){
    QByteArray cmdOut = processVis->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        if(logs.contains("finished")){
            terminal->print("可视化已完成！");
            if(processVis->state()==QProcess::Running){
                processVis->close();
                processVis->kill();
            }
            ui->progressBar_mV_visFea->setMaximum(100);
            ui->progressBar_mV_visFea->setValue(100);
            // 按页加载图像
            this->currFeaPage = 0;
            // 获取图片文件夹下的所有图片文件名
            vector<string> imageFileNames;
            dirTools->getFiles(imageFileNames, ".png", this->feaImgsSavePath.toStdString());
            this->feaNum = imageFileNames.size();
            this->allFeaPage = this->feaNum/NUM_FEA + bool(this->feaNum%NUM_FEA);
            ui->label_mV_pageAll->setText(QString::number(this->allFeaPage));
            nextFeaImgsPage();
        }
        if(logs.contains("Error") || logs.contains("Errno")){
            terminal->print("可视化失败！");
            QMessageBox::warning(NULL,"错误","所选隐层不支持可视化!");
            ui->progressBar_mV_visFea->setMaximum(100);
            ui->progressBar_mV_visFea->setValue(0);
        }
    }
}


void ModelVisPage::nextFeaImgsPage(){
    if(this->feaNum==0){
        QMessageBox::warning(NULL,"错误","未进行可视化!");
        return;
    }
    // 当前页码更新
    this->currFeaPage = (this->currFeaPage%this->allFeaPage)+1;
    ui->label_mV_pageCurr->setText(QString::number(this->currFeaPage));
    
    if(this->feaNum==1){    // 全连接层及其他一维可视化
        if(this->actOrGrad == 0){
            recvShowPicSignal(QPixmap(this->feaImgsSavePath+"/"+"FC_vis.png"), ui->graphicsView_mV_fea1);
            ui->label_mV_fea1->setText("神经元激活分布");
        }
        else{
            recvShowPicSignal(QPixmap(this->feaImgsSavePath+"/"+"FC_vis.png"), ui->graphicsView_mV_fea2);
            ui->label_mV_fea2->setText("神经元梯度分布");
        }
    }
    else{                   // 卷积层及其他二维可视化    
        // 当前页所要展示的特征图索引
        int beginIdx = NUM_FEA*(this->currFeaPage-1)+1;
        int endIdx = NUM_FEA*this->currFeaPage;
        if(endIdx>this->feaNum){
            endIdx = this->feaNum;
        }
        vector<int> showFeaIndex(endIdx-beginIdx+1);
        std::iota(showFeaIndex.begin(), showFeaIndex.end(), beginIdx);

        // 加载特征图和标签,并显示
        for(size_t i = 0; i<showFeaIndex.size(); i++){
            QGraphicsView *feaGraphic = ui->groupBox_mV_feaShow->findChild<QGraphicsView *>("graphicsView_mV_fea"+QString::number(i+1));
            QLabel *fealabel = ui->groupBox_mV_feaShow->findChild<QLabel *>("label_mV_fea"+QString::number(i+1));
            recvShowPicSignal(QPixmap(this->feaImgsSavePath+"/"+QString::number(showFeaIndex[i])+".png"), feaGraphic);
            fealabel->setText("CH-"+QString::number(showFeaIndex[i]));
        }
    }
}


void ModelVisPage::refreshGlobalInfo(){
    // 数据集信息更新
    if(datasetInfo->checkMap(datasetInfo->selectedType, datasetInfo->selectedName, "PATH")){
        ui->label_mV_dataset->setText(QString::fromStdString(datasetInfo->selectedName));
        this->choicedDatasetPATH = datasetInfo->getAttri(datasetInfo->selectedType, datasetInfo->selectedName, "PATH");
    }
    else{
        this->choicedDatasetPATH = "";
        ui->label_mV_dataset->setText("空");
    }
    // 模型信息更新
    if(!modelInfo->selectedName.empty() && modelInfo->checkMap(modelInfo->selectedType, modelInfo->selectedName, "PATH")){
        ui->label_mV_model->setText(QString::fromStdString(modelInfo->selectedName));
        this->choicedModelPATH = modelInfo->getAttri(modelInfo->selectedType, modelInfo->selectedName, "PATH");
    }
    else{
        this->choicedModelPATH = "";
        ui->label_mV_model->setText(QString::fromStdString("空"));
    }

    // 模型类型，目前可视化仅支持.hdf5和.pth
    this->choicedModelSuffix = "." + QString::fromStdString(this->choicedModelPATH).split(".").last();
    QString modelBasePath = "";
    if(this->choicedModelSuffix==".pth" || this->choicedModelSuffix==".hdf5"){
        modelBasePath = QString::fromStdString(this->choicedModelPATH).split(choicedModelSuffix).first();
    }
    else{        // 不支持的模型类型
        this->choicedModelPATH = "";
    }

    this->modelCheckpointPath   = QString::fromStdString(this->choicedModelPATH);
    this->modelStructXmlPath    = modelBasePath.toStdString() + "_struct.xml";
    this->modelStructImgPath    = modelBasePath + "_structImage";
    this->feaImgsSavePath       = modelBasePath + "_modelVisOutput";

    clearComboBox();
}


void ModelVisPage::clearComboBox(){
    // 判断是否存在模型结构文件*_struct.xml，如果没有则返回
    if (!dirTools->exist(this->modelStructXmlPath)){
        ui->comboBox_mV_L1->clear();
        ui->comboBox_mV_L2->clear();
        ui->comboBox_mV_L3->clear();
        ui->comboBox_mV_L4->clear();
        ui->comboBox_mV_L5->clear();
        return;
    } 
    // 判断是否存在可视化输出文件夹，如果没有则创建
    QDir dir(this->feaImgsSavePath);
    if(!dir.exists()){
        dir.mkpath(this->feaImgsSavePath);
    }
    // 初始化第一个下拉框
    QStringList L1Layers;
    loadModelStruct_L1(L1Layers);
    ui->comboBox_mV_L1->clear();
    ui->comboBox_mV_L1->addItems(L1Layers);
    ui->comboBox_mV_L2->clear();
    ui->comboBox_mV_L3->clear();
    ui->comboBox_mV_L4->clear();
    ui->comboBox_mV_L5->clear();

    this->choicedLayer["L1"] = "NULL";
    this->choicedLayer["L2"] = "NULL";
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    refreshVisInfo();
    // }
}


void ModelVisPage::refreshVisInfo(){
    // 提取目标层信息的特定格式
    QString targetVisLayer = "";
    if(this->choicedModelSuffix==".pth"){
        vector<string> tmpList = {"L2", "L3"};
        for(auto &layer : tmpList){
            if(this->choicedLayer[layer] == "NULL"){
                continue;
            }
            if(layer == "L2"){
                targetVisLayer += QString::fromStdString(this->choicedLayer[layer]);
            }
            else{
                if(this->choicedLayer[layer][0] == '_'){
                    targetVisLayer += QString::fromStdString("["+this->choicedLayer[layer].substr(1)+"]");
                }
                else{
                    targetVisLayer += QString::fromStdString("."+this->choicedLayer[layer]);
                }
            }
        }
        this->targetVisLayer = targetVisLayer.replace("._", ".");
    }
    else if(this->choicedModelSuffix==".hdf5"){
        vector<string> tmpList = {"L2", "L3", "L4", "L5"};
        for(auto &layer : tmpList){
            if(this->choicedLayer[layer] == "NULL"){
                continue;
            }
            if(layer == "L2"){
                targetVisLayer += QString::fromStdString(this->choicedLayer[layer]);
            }
            else{
                targetVisLayer += QString::fromStdString("_"+this->choicedLayer[layer]);
            }
        }
        this->targetVisLayer = targetVisLayer.replace("__", "_");
    }

    ui->label_mV_visLayer->setText(this->targetVisLayer);

    // 加载相应的预览图像
    QString imgPath = this->modelStructImgPath + "/";
    if(this->targetVisLayer == ""){
        imgPath += "framework.png";
    }
    else{
        imgPath = imgPath + this->targetVisLayer + ".png"; 
    }
    if(this->dirTools->exist(imgPath.toStdString())){
        recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_mV_modelImg);
    }
}


int ModelVisPage::randomImage(){
    if(this->choicedDatasetPATH.empty() || this->choicedModelPATH.empty()){
        QMessageBox::warning(NULL,"错误","未选择数据集/模型!");
        return -1;
    }
    // 目前仅支持HRRP模型
    if(datasetInfo->selectedType!="HRRP"){
        QMessageBox::warning(NULL,"错误","目前仅支持HRRP数据集!");
        return -1;
    }
    // 获取所有子文件夹
    vector<string> allSubDirs;
    dirTools->getDirs(allSubDirs, choicedDatasetPATH);
    string choicedSubDir = allSubDirs[(rand())%allSubDirs.size()];
    string classPath = choicedDatasetPATH + "/" + choicedSubDir;

    // if(this->choicedModelSuffix == ".pth"){
        // vector<string> txtFileNames;
        // if(dirTools->getFiles(txtFileNames, ".txt", classPath)){
        //     // 随机选取一个信号作为预览
        //     srand((unsigned)time(NULL));
        //     string choicedTxtFile = txtFileNames[(rand())%txtFileNames.size()];
        //     string choicedTxtPath = classPath+"/"+choicedTxtFile;
        //     this->choicedSamplePATH = QString::fromStdString(choicedTxtPath);

        //     // 绘制样本
        //     // qDebug()<<choicedSamplePATH;
        //     Chart *previewChart = new Chart(ui->label_mV_choicedImg,"HRRP(Ephi),Polarization HP(1)[Magnitude in dB]",choicedSamplePATH);
        //     previewChart->drawHRRPimage(ui->label_mV_choicedImg);
        //     ui->label_mV_choicedImgName->setText(QString::fromStdString(choicedSubDir)+"/"+choicedSamplePATH.split("/").last());
        //     return 1;
        // }
        // return -1;
    // }
    if (this->choicedModelSuffix == ".hdf5" || this->choicedModelSuffix == ".pth"){
        vector<string> allMatFile;
        if(dirTools->getFiles(allMatFile, ".mat", classPath)){
            QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
            //下面这部分代码都是为了让randomIdx在合理的范围内
            MATFile* pMatFile = NULL;
            mxArray* pMxArray = NULL;
            pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
            if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空!";return -1;}
            std::string matVariable=allMatFile[0].substr(0,allMatFile[0].find_last_of('.')).c_str();//假设数据变量名同文件名的话

            pMxArray = matGetVariable(pMatFile,matVariable.c_str());
            if(!pMxArray){qDebug()<<"(ModelEvalPage::randSample)pMxArray变量没找到!";return -1;}
            int N = mxGetN(pMxArray);  //N 列数
            int randomIdx = N-(rand())%N;

            // 保存随机选取的样本路径
            this->choicedSamplePATH = matFilePath;
            this->choicedMatIdx = randomIdx;

            //绘图
            QString chartTitle="Temporary Title";
            if(datasetInfo->selectedType=="HRRP") {chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";}
            Chart *previewChart = new Chart(ui->label_mV_choicedImg,chartTitle,matFilePath);
            previewChart->drawImage(ui->label_mV_choicedImg,datasetInfo->selectedType,randomIdx);
            ui->label_mV_choicedImgName->setText(QString::fromStdString(allMatFile[0])+"/"+QString::number(randomIdx));

            return 1;
        }
        return -1;
    }
    else{
        return -1;
    }

}


int ModelVisPage::importImage(){
    QString filePath = QFileDialog::getOpenFileName(NULL, "导入数据", "./", "Txt Files (*.txt)");
    if(filePath.isEmpty()){
        return -1;
    }
    this->choicedSamplePATH = filePath;
//    recvShowPicSignal(QPixmap(filePath), ui->graphicsView_mV_choicedImg);
    ui->label_mV_choicedImgName->setText(choicedSamplePATH.split("/").last());

    return 1;
}


void ModelVisPage::on_comboBox_L1(QString choicedLayer){
    this->choicedLayer["L1"] = choicedLayer.toStdString();
    this->choicedLayer["L2"] = "NULL";
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L2(nextLayers);
    ui->comboBox_mV_L2->clear();
    ui->comboBox_mV_L2->addItems(nextLayers);
    ui->comboBox_mV_L3->clear();
    ui->comboBox_mV_L4->clear();
    ui->comboBox_mV_L5->clear();
    refreshVisInfo();
}

void ModelVisPage::on_comboBox_L2(QString choicedLayer){
    this->choicedLayer["L2"] = choicedLayer.toStdString();
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L3(nextLayers);
    ui->comboBox_mV_L3->clear();
    ui->comboBox_mV_L3->addItems(nextLayers);
    ui->comboBox_mV_L4->clear();
    ui->comboBox_mV_L5->clear();
    refreshVisInfo();
}

void ModelVisPage::on_comboBox_L3(QString choicedLayer){
    this->choicedLayer["L3"] = choicedLayer.toStdString();
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L4(nextLayers);
    ui->comboBox_mV_L4->clear();
    ui->comboBox_mV_L4->addItems(nextLayers);
    ui->comboBox_mV_L5->clear();
    refreshVisInfo();
}

void ModelVisPage::on_comboBox_L4(QString choicedLayer){
    this->choicedLayer["L4"] = choicedLayer.toStdString();
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L5(nextLayers);
    ui->comboBox_mV_L5->clear();
    ui->comboBox_mV_L5->addItems(nextLayers);
    refreshVisInfo();
}

void ModelVisPage::on_comboBox_L5(QString choicedLayer){
    this->choicedLayer["L5"] = choicedLayer.toStdString();
    refreshVisInfo();
}



void ModelVisPage::loadModelStruct_L1(QStringList &currLayers){
    TiXmlDocument datasetInfoDoc(this->modelStructXmlPath.c_str());    //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the modelStruct .xml file. Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info
    //遍历一级根结点
    for(TiXmlElement *currL1Ele = RootElement->FirstChildElement(); currL1Ele != NULL; currL1Ele = currL1Ele->NextSiblingElement()){
        // cout<<"----->"<<currL1Ele->Value()<<endl;
        currLayers.append(QString::fromStdString(currL1Ele->Value()));
    }
}

void ModelVisPage::loadModelStruct_L2(QStringList &currLayers){
    TiXmlDocument datasetInfoDoc(this->modelStructXmlPath.c_str());    //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the modelStruct .xml file. Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info
    //遍历一级根结点
    for(TiXmlElement *currL1Ele = RootElement->FirstChildElement(); currL1Ele != NULL; currL1Ele = currL1Ele->NextSiblingElement()){
        if(currL1Ele->Value() == this->choicedLayer["L1"]){
            // 遍历二级子节点
            for(TiXmlElement *currL2Ele=currL1Ele->FirstChildElement(); currL2Ele != NULL; currL2Ele=currL2Ele->NextSiblingElement()){
                // cout<<"---->"<<currL2Ele->Value()<<endl;
                currLayers.append(QString::fromStdString(currL2Ele->Value()));
            }
        }

    }
}

void ModelVisPage::loadModelStruct_L3(QStringList &currLayers){
    TiXmlDocument datasetInfoDoc(this->modelStructXmlPath.c_str());    //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the modelStruct .xml file. Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info
    //遍历一级根结点
    for(TiXmlElement *currL1Ele = RootElement->FirstChildElement(); currL1Ele != NULL; currL1Ele = currL1Ele->NextSiblingElement()){
        if(currL1Ele->Value() == this->choicedLayer["L1"]){
            // 遍历二级子节点
            for(TiXmlElement *currL2Ele=currL1Ele->FirstChildElement(); currL2Ele != NULL; currL2Ele=currL2Ele->NextSiblingElement()){
                if(currL2Ele->Value() == this->choicedLayer["L2"]){
                    // 遍历三级子节点
                    for(TiXmlElement *currL3Ele=currL2Ele->FirstChildElement(); currL3Ele != NULL; currL3Ele=currL3Ele->NextSiblingElement()){
                        // cout<<"---->"<<currL3Ele->Value()<<endl;
                        currLayers.append(QString::fromStdString(currL3Ele->Value()));
                    }
                }
            }
        }

    }
}

void ModelVisPage::loadModelStruct_L4(QStringList &currLayers){
    TiXmlDocument datasetInfoDoc(this->modelStructXmlPath.c_str());    //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the modelStruct .xml file. Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info
    //遍历一级根结点
    for(TiXmlElement *currL1Ele = RootElement->FirstChildElement(); currL1Ele != NULL; currL1Ele = currL1Ele->NextSiblingElement()){
        if(currL1Ele->Value() == this->choicedLayer["L1"]){
            // 遍历二级子节点
            for(TiXmlElement *currL2Ele=currL1Ele->FirstChildElement(); currL2Ele != NULL; currL2Ele=currL2Ele->NextSiblingElement()){
                if(currL2Ele->Value() == this->choicedLayer["L2"]){
                    // 遍历三级子节点
                    for(TiXmlElement *currL3Ele=currL2Ele->FirstChildElement(); currL3Ele != NULL; currL3Ele=currL3Ele->NextSiblingElement()){
                        if(currL3Ele->Value() == this->choicedLayer["L3"]){
                        // 遍历四级子节点
                            for(TiXmlElement *currL4Ele=currL3Ele->FirstChildElement(); currL4Ele != NULL; currL4Ele=currL4Ele->NextSiblingElement()){
                                // cout<<"---->"<<currL4Ele->Value()<<endl;
                                currLayers.append(QString::fromStdString(currL4Ele->Value()));
                            }
                        }
                    }
                }
            }
        }
    }
}

void ModelVisPage::loadModelStruct_L5(QStringList &currLayers){
    TiXmlDocument datasetInfoDoc(this->modelStructXmlPath.c_str());    //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the modelStruct .xml file. Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info
    //遍历一级根结点
    for(TiXmlElement *currL1Ele = RootElement->FirstChildElement(); currL1Ele != NULL; currL1Ele = currL1Ele->NextSiblingElement()){
        if(currL1Ele->Value() == this->choicedLayer["L1"]){
            // 遍历二级子节点
            for(TiXmlElement *currL2Ele=currL1Ele->FirstChildElement(); currL2Ele != NULL; currL2Ele=currL2Ele->NextSiblingElement()){
                if(currL2Ele->Value() == this->choicedLayer["L2"]){
                    // 遍历三级子节点
                    for(TiXmlElement *currL3Ele=currL2Ele->FirstChildElement(); currL3Ele != NULL; currL3Ele=currL3Ele->NextSiblingElement()){
                        if(currL3Ele->Value() == this->choicedLayer["L3"]){
                        // 遍历四级子节点
                            for(TiXmlElement *currL4Ele=currL3Ele->FirstChildElement(); currL4Ele != NULL; currL4Ele=currL4Ele->NextSiblingElement()){
                                if(currL4Ele->Value() == this->choicedLayer["L4"]){
                                    // 遍历五级子节点
                                    for(TiXmlElement *currL5Ele=currL4Ele->FirstChildElement(); currL5Ele != NULL; currL5Ele=currL5Ele->NextSiblingElement()){
                                        // cout<<"---->"<<currL5Ele->Value()<<endl;
                                        currLayers.append(QString::fromStdString(currL5Ele->Value()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void ModelVisPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image);  //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.95;              //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.95;           //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}
