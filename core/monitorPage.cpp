#include "monitorPage.h"
#include "qimagereader.h"
#include <iostream>
#include <QMessageBox>
#include <QMutex>

MonitorPage::MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo)
{
    label2class[0] ="bigball";label2class[1] ="DT"; label2class[2] ="Moxiu";
    label2class[3] ="sallball"; label2class[4] ="taper"; label2class[5] ="WD";
    for(auto &item: label2class){
        class2label[item.second] = item.first;
    }

    QSemaphore sem;
    QMutex lock;
    inferThread =new InferThread(&sem,&sharedQue,&lock);//推理线程
    inferThread->setInferMode("real_time_infer");
    //connect(inferThread, &InferThread::sigInferResult,this,&MonitorPage::showInferResult);
    connect(inferThread, SIGNAL(sigInferResult(int,QVariant)),this,SLOT(showInferResult(int,QVariant)));
    
    server = new SocketServer(&sem,&sharedQue,&lock,terminal);//监听线程
    connect(server, SIGNAL(sigColorMap()),this,SLOT(showColorMap()));

    client = new SocketClient();
    connect(client, SIGNAL(sigClassName(int)),this,SLOT(showRealClass(int)));

    connect(ui->startListen, &QPushButton::clicked, this, &MonitorPage::startListen);
    connect(ui->simulateSignal, &QPushButton::clicked, this, &MonitorPage::simulateSend);
    connect(ui->stopListen, &QPushButton::clicked,[this]() { delete server; });

}
void MonitorPage::startListen(){
    if(modelInfo->selectedType==""||this->choicedDatasetPATH==""){
        QMessageBox::warning(NULL, "实时监测", "监听失败,请先指定HRRP模型和数据集");
        qDebug()<<"modelInfo->selectedType=="<<QString::fromStdString(modelInfo->selectedType);
        return;
    }
    server->start();
    terminal->print("开始监听");
    inferThread->start();
}

void MonitorPage::simulateSend(){
    client->start();
}

void MonitorPage::refresh(){
    bool ifDataPreProcess=true;
    // 网络输出标签对应类别名称初始化
    std::vector<std::string> comboBoxContents = datasetInfo->selectedClassNames;
    if(comboBoxContents.size()>0){
        for(int i=0;i<comboBoxContents.size();i++)   label2class[i]=comboBoxContents[i];
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    //如果数据集或模型路径变了
    if(modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH")!=this->choicedModelPATH ||
    this->choicedDatasetPATH != datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH")){
        if(datasetInfo->selectedType=="INCRE") ifDataPreProcess=false;
        //trtInfer = new TrtInfer(class2label);
        this->choicedModelPATH=modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
        this->choicedDatasetPATH=datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
        inferThread->setClass2LabelMap(class2label);
        //qDebug()<<"(MonitorPage::refresh) class2label.size()=="<<class2label.size();
        inferThread->setParmOfRTI(this->choicedModelPATH,ifDataPreProcess);//只有小样本是false 既不做预处理
        client->setClass2LabelMap(class2label);
        client->setParmOfRTI(this->choicedDatasetPATH,ifDataPreProcess);
    }
}

void removeLayout2(QLayout *layout){
    QLayoutItem *child;
    if (layout == nullptr)
        return;
    while ((child = layout->takeAt(0)) != nullptr){
        // child可能为QLayoutWidget、QLayout、QSpaceItem
        // QLayout、QSpaceItem直接删除、QLayoutWidget需删除内部widget和自身
        if (QWidget* widget = child->widget()){
            widget->setParent(nullptr);
            delete widget;
            widget = nullptr;
        }
        else if (QLayout* childLayout = child->layout())
            removeLayout2(childLayout);
        delete child;
        child = nullptr;
    }
}

void MonitorPage::showInferResult(int predIdx,QVariant qv){
    Chart *tempChart = new Chart(ui->label_mE_chartGT,"","");//就调用一下它的方法
    //std::vector<float> degrees={0.1,0.1,0.1,0.1,0.2,0.4};
    std::vector<float> degrees=qv.value<std::vector<float>>();
    QString predClass = QString::fromStdString(label2class[predIdx]);
    //terminal->print("Real-time classification results:"+predClass);//连续调用恐怕会有问题
    QWidget *tempWidget=tempChart->drawDisDegreeChart(predClass,degrees,label2class);
    //QWidget *tempWidget2=tempChart->drawDisDegreeChart(predClass,degrees,label2class);
    removeLayout2(ui->horizontalLayout_degreeChart2);
    ui->horizontalLayout_degreeChart2->addWidget(tempWidget);
    ui->jcLabel->setText(QString::fromStdString(label2class[predIdx]));
}

void MonitorPage::showColorMap(){
    /*=================draw thermal column==============*/
    QLabel *imageLabel=new QLabel;
    imageLabel->setBackgroundRole(QPalette::Base);
    imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel->setScaledContents(true);
    //imageLabel->setStyleSheet("border:2px solid red;");
    QImage image;
    QImageReader reader("D:/colorMap.png");
    reader.setAutoTransform(true);
    const QImage newImage = reader.read();
    if (newImage.isNull()) {
        qDebug()<<"errrrrrrrrrror";
    }
    image = newImage;
    imageLabel->setPixmap(QPixmap::fromImage(image));
    removeLayout2(ui->horizontalLayout_HotCol);
    ui->horizontalLayout_HotCol->addWidget(imageLabel);
}

void MonitorPage::showRealClass(int realLabel){//client触发
    ui->xlLabel->setText(QString::fromStdString(label2class[realLabel]));
}

MonitorPage::~MonitorPage(){

}
