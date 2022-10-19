#include "modelCAMPage.h"

#include "./lib/guiLogic/tinyXml/tinyxml.h"
#include "./lib/guiLogic/tools/convertTools.h"
#include <QGraphicsScene>
#include <QMessageBox>
#include <QFileDialog>

using namespace std;


ModelCAMPage::ModelCAMPage(Ui_MainWindow *main_ui,
                             BashTerminal *bash_terminal,
                             DatasetInfo *globalDatasetInfo,
                             ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo)
{   
    // 刷新模型、数据集信息
    refreshGlobalInfo();
    this->condaPath = "/home/z840/anaconda3/bin/activate";
    this->condaEnvName = "mmlab";

    // 下拉框信号槽绑定
    connect(ui->comboBox_CAM_L1, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L1(QString)));
    connect(ui->comboBox_CAM_L2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L2(QString)));
    connect(ui->comboBox_CAM_L3, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L3(QString)));
    connect(ui->comboBox_CAM_L4, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L4(QString)));
    connect(ui->comboBox_CAM_L5, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L5(QString)));
    connect(ui->comboBox_CAM_method, SIGNAL(textActivated(QString)), this, SLOT(showCamFig(QString)));

    // 按钮信号槽绑定
    connect(ui->pushButton_CAM_clear, &QPushButton::clicked, this, &ModelCAMPage::clearComboBox);
    connect(ui->pushButton_CAM_randomImg, &QPushButton::clicked, this, &ModelCAMPage::randomImage);
    connect(ui->pushButton_CAM_importImg, &QPushButton::clicked, this, &ModelCAMPage::importImage);
    connect(ui->pushButton_CAM_confirmVis, &QPushButton::clicked, this, &ModelCAMPage::confirmVis);
    // 多线程的信号槽绑定
    processVis = new QProcess();
    connect(processVis, &QProcess::readyReadStandardOutput, this, &ModelCAMPage::processVisFinished);

    // 所支持的可视化方法初始化
    QStringList allCamMethods = {
        "gradcam", "gradcam++", "xgradcam", "eigengradcam", "layercam",
        "ablationcam", "eigencam", "featmapam",
    };
    ui->comboBox_CAM_method->addItems(allCamMethods);

}

ModelCAMPage::~ModelCAMPage(){

}


void ModelCAMPage::confirmVis(){
    this->choicedCamMethod = ui->comboBox_CAM_method->currentText();
    if(this->choicedSamplePATH.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择输入图像!");
        return;
    }
    if(this->modelCheckpointPath.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择要可视化模型!");
        return;
    }
    if(this->modelConfigPath.isEmpty()){
        QMessageBox::warning(NULL,"错误","选择的模型没有配置文件，不支持可视化!");
        return;
    }
    if(this->targetVisLayer.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择可视化位置!");
        return;
    }
    if(choicedCamMethod.isEmpty()){
        QMessageBox::warning(NULL,"错误","未指定可视化CAM方法!");
        return;
    }
    // QString output;
    // 激活conda python环境
    QString activateEnv = "source "+this->condaPath+" "+this->condaEnvName+"&&";
    QString command = activateEnv + \
        "python "+"../api/modelVis/vis_cam.py"+ \ 
        " --img="           +this->choicedSamplePATH+ \
        " --config="        +this->modelConfigPath+ \
        " --checkpoint="    +this->modelCheckpointPath+ \
        " --target-layers=" +this->targetVisLayer+ \ 
        " --out-dir="       +this->camImgsSavePath+ \
        " --method="        +this->choicedCamMethod;
    // 执行python脚本
    this->terminal->print(command);
    this->execuCmdProcess(command);
}


// 可视化线程
void ModelCAMPage::execuCmdProcess(QString cmd){
    if(processVis->state()==QProcess::Running){
        processVis->close();
        processVis->kill();
    }
    processVis->setProcessChannelMode(QProcess::MergedChannels);
    processVis->start(this->terminal->bashApi);
    ui->progressBar_CAM->setMaximum(0);
    ui->progressBar_CAM->setValue(0);
    processVis->write(cmd.toLocal8Bit() + '\n');
}


void ModelCAMPage::processVisFinished(){
    QByteArray cmdOut = processVis->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        if(logs.contains("finished")){
            terminal->print("可视化已完成！");
            if(processVis->state()==QProcess::Running){
                processVis->close();
                processVis->kill();
            }
            ui->progressBar_CAM->setMaximum(100);
            ui->progressBar_CAM->setValue(100);

            // 加载图像
            ui->label_CAM_camImgLabel->setText(this->choicedCamMethod);
            recvShowPicSignal(QPixmap(this->camImgsSavePath+"/CAM_output.png"), ui->graphicsView_CAM_camImg);

        }
        if(logs.contains("Error")){
            terminal->print("可视化失败！");
            QMessageBox::warning(NULL,"错误","所选隐层不支持决策可视化!");
            ui->progressBar_CAM->setMaximum(100);
            ui->progressBar_CAM->setValue(0);
        }
    }
}

void ModelCAMPage::refreshGlobalInfo(){
    // 基本信息更新
    if(datasetInfo->checkMap(datasetInfo->selectedType, datasetInfo->selectedName, "PATH")){
        this->choicedDatasetPATH = datasetInfo->getAttri(datasetInfo->selectedType, datasetInfo->selectedName, "PATH");
        ui->label_CAM_dataset->setText(QString::fromStdString(datasetInfo->selectedName));
    }
    else{
        this->choicedDatasetPATH = "";
        ui->label_CAM_dataset->setText("空");
    }
    if(!modelInfo->selectedName.empty() && modelInfo->checkMap(modelInfo->selectedType, modelInfo->selectedName, "PATH")){
        ui->label_CAM_model->setText( QString::fromStdString(modelInfo->selectedName));
        // python可解释接口所需要的路径更新
        QString choicedModelPath = QString::fromStdString(modelInfo->getAttri(modelInfo->selectedType, modelInfo->selectedName, "PATH"));
        QString modelBasePath = choicedModelPath.split(".mar").first();
        this->modelStructXmlPath    = modelBasePath.toStdString() + "_struct.xml";
        this->modelStructImgPath    = modelBasePath + "_structImage";
        this->modelConfigPath       = modelBasePath + "_config.py";
        this->modelCheckpointPath   = modelBasePath + "_checkpoint.pth";

        this->camImgsSavePath       = modelBasePath + "_modelCAMOutput";

        clearComboBox();
    }
    else{
        ui->label_CAM_model->setText( QString::fromStdString("空"));
    }
}


void ModelCAMPage::clearComboBox(){
    if(!modelInfo->selectedName.empty()){
        // 初始化第一个下拉框
        QStringList L1Layers;
        loadModelStruct_L1(L1Layers);
        ui->comboBox_CAM_L1->clear();
        ui->comboBox_CAM_L1->addItems(L1Layers);
        ui->comboBox_CAM_L2->clear();
        ui->comboBox_CAM_L3->clear();
        ui->comboBox_CAM_L4->clear();
        ui->comboBox_CAM_L5->clear();

        this->choicedLayer["L1"] = "NULL";
        this->choicedLayer["L2"] = "NULL";
        this->choicedLayer["L3"] = "NULL";
        this->choicedLayer["L4"] = "NULL";
        this->choicedLayer["L5"] = "NULL";

        refreshVisInfo();
    }
}


void ModelCAMPage::refreshVisInfo(){
    // 提取目标层信息的特定格式
    QString targetVisLayer = "";
    vector<string> tmpList = {"L1", "L2", "L3", "L4", "L5"};
    for(auto &layer : tmpList){
        if(this->choicedLayer[layer] == "NULL"){
            continue;
        }
        if(layer == "L1"){
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
    ui->label_CAM_visLayer->setText(this->targetVisLayer);

    // 加载相应的预览图像
    QString imgPath = this->modelStructImgPath + "/";
    if(this->targetVisLayer == ""){
        imgPath += "framework.png";
    }
    else{
        imgPath = imgPath + this->targetVisLayer + ".png"; 
    }
    if(this->dirTools->exist(imgPath.toStdString())){
        recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_CAM_modelImg);
    }
}


int ModelCAMPage::randomImage(){
    if(this->choicedDatasetPATH.empty()){
        QMessageBox::warning(NULL,"错误","未选择数据集!");
        return -1;
    }
    // 获取所有子文件夹，并判断是否是图片、标注文件夹
    vector<string> allSubDirs;
    dirTools->getDirs(allSubDirs, choicedDatasetPATH);
    vector<string> targetKeys = {"images","labelTxt"};
    for (auto &targetKey: targetKeys){
        if(!(std::find(allSubDirs.begin(), allSubDirs.end(), targetKey) != allSubDirs.end())){
            // 目标路径不存在目标文件夹
            QMessageBox::warning(NULL,"错误","该数据集路径下不存在"+QString::fromStdString(targetKey)+"文件夹！");
            return -1;
        }
    }
    // 获取图片文件夹下的所有图片文件名
    vector<string> imageFileNames;
    dirTools->getFiles(imageFileNames, ".png", choicedDatasetPATH+"/images");
    
    // 随机选取一张图片作为预览图片
    srand((unsigned)time(NULL));
    string choicedImageFile = imageFileNames[(rand())%imageFileNames.size()];
    string choicedImagePath = choicedDatasetPATH+"/images/"+choicedImageFile;
    this->choicedSamplePATH = QString::fromStdString(choicedImagePath);

    // 读取图片
    cv::Mat imgSrc = cv::imread(choicedImagePath.c_str(), cv::IMREAD_COLOR);

    // 读取GroundTruth，包含四个坐标和类别信息
    std::vector<std::vector<cv::Point>> points_GT;
    std::vector<std::string> labels_GT;
    string labelPath = choicedDatasetPATH+"/labelTxt/"+choicedImageFile.substr(0,choicedImageFile.size()-4)+".txt";
    dirTools->getGroundTruth(labels_GT, points_GT, labelPath);

    // 在图片上画出GroundTruth的矩形框
    cv::drawContours(imgSrc, points_GT, -1, cv::Scalar(16, 124, 16), 2);
    // 绘制类别标签到图片上
    for(size_t i = 0; i<labels_GT.size(); i++){
        cv::putText(imgSrc, labels_GT[i], points_GT[i][1], cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 204, 0), 1);
    }
    // 将图片显示到界面上
    recvShowPicSignal(CVS::cvMatToQPixmap(imgSrc), ui->graphicsView_CAM_choicedImg);
    ui->label_CAM_choicedImgName->setText(choicedSamplePATH.split("/").last());
}


int ModelCAMPage::importImage(){
    QString filePath = QFileDialog::getOpenFileName(NULL, "导入图片", "./", "Image Files (*.png *.jpg *.bmp *.tiff *.raw)");
    if(filePath.isEmpty()){
        return -1;
    }
    this->choicedSamplePATH = filePath;
    recvShowPicSignal(QPixmap(filePath), ui->graphicsView_CAM_choicedImg);
    ui->label_CAM_choicedImgName->setText(choicedSamplePATH.split("/").last());
}


void ModelCAMPage::on_comboBox_L1(QString choicedLayer){
    this->choicedLayer["L1"] = choicedLayer.toStdString();
    this->choicedLayer["L2"] = "NULL";
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L2(nextLayers);
    ui->comboBox_CAM_L2->clear();
    ui->comboBox_CAM_L2->addItems(nextLayers);
    ui->comboBox_CAM_L3->clear();
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L5->clear();
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L2(QString choicedLayer){
    this->choicedLayer["L2"] = choicedLayer.toStdString();
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L3(nextLayers);
    ui->comboBox_CAM_L3->clear();
    ui->comboBox_CAM_L3->addItems(nextLayers);
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L5->clear();
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L3(QString choicedLayer){
    this->choicedLayer["L3"] = choicedLayer.toStdString();
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L4(nextLayers);
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L4->addItems(nextLayers);
    ui->comboBox_CAM_L5->clear();
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L4(QString choicedLayer){
    this->choicedLayer["L4"] = choicedLayer.toStdString();
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L5(nextLayers);
    ui->comboBox_CAM_L5->clear();
    ui->comboBox_CAM_L5->addItems(nextLayers);
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L5(QString choicedLayer){
    this->choicedLayer["L5"] = choicedLayer.toStdString();
    refreshVisInfo();
}


void ModelCAMPage::showCamFig(QString method){
    // TODO
}


void ModelCAMPage::loadModelStruct_L1(QStringList &currLayers){
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

void ModelCAMPage::loadModelStruct_L2(QStringList &currLayers){
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

void ModelCAMPage::loadModelStruct_L3(QStringList &currLayers){
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

void ModelCAMPage::loadModelStruct_L4(QStringList &currLayers){
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

void ModelCAMPage::loadModelStruct_L5(QStringList &currLayers){
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


void ModelCAMPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
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
