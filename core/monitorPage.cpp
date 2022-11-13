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
    //初始化label2class缓解了acquire在没有release的情况下就成功的问题
    label2class[0] ="Big_ball";label2class[1] ="Cone"; label2class[2] ="Cone_cylinder";
    label2class[3] ="DT"; label2class[4] ="Small_ball"; label2class[5] ="Spherical_cone";
    for(auto &item: label2class){
        class2label[item.second] = item.first;
    }
    ui->simulateSignal->setEnabled(false);
    ui->stopListen->setEnabled(false);
    QSemaphore sem(0);
    QMutex lock;
    inferThread =new InferThread(&sem,&sharedQue,&lock);//推理线程
    inferThread->setInferMode("real_time_infer");
    //connect(inferThread, &InferThread::sigInferResult,this,&MonitorPage::showInferResult);
    connect(inferThread, SIGNAL(sigInferResult(int,QVariant)),this,SLOT(showInferResult(int,QVariant)));
    connect(inferThread, SIGNAL(modelAlready()),this,SLOT(enableSimulateSignal()));

    server = new SocketServer(&sem,&sharedQue,&lock,terminal);//监听线程
    connect(server, SIGNAL(sigColorMap()),this,SLOT(showColorMap()));

    client = new SocketClient();
    connect(client, SIGNAL(sigClassName(int)),this,SLOT(showRealClass(int)));

    connect(ui->startListen, &QPushButton::clicked, this, &MonitorPage::startListen);
    connect(ui->simulateSignal, &QPushButton::clicked, this, &MonitorPage::simulateSend);
    connect(ui->stopListen, &QPushButton::clicked,[this]() { 
        // ui->simulateSignal->setEnabled(false);
        // ui->stopListen->setEnabled(false);
        stopSend();
        // delete client; 
        // delete server; 
        //delete inferThread;
    });
    //connect(ui->stopListen, &QPushButton::clicked, this, &MonitorPage::stopListen);
    connect(this, SIGNAL(startOrstop_sig(bool)), client, SLOT(startOrstop_slot(bool)));

}

void MonitorPage::startListen(){
    if(modelInfo->selectedType==""||this->choicedDatasetPATH==""){
        QMessageBox::warning(NULL, "实时监测", "监听失败,请先指定HRRP模型和数据集");
        //qDebug()<<"modelInfo->selectedType=="<<QString::fromStdString(modelInfo->selectedType);
        return;
    }
    server->start();
    terminal->print("开始监听");
    inferThread->start();
    emit startOrstop_sig(true);
}

void MonitorPage::stopSend(){
    emit startOrstop_sig(false);
    //client->StopThread();
}

void MonitorPage::simulateSend(){
    client->start();
    ui->stopListen->setEnabled(true);
}

void MonitorPage::refresh(){
    bool ifDataPreProcess=true;
    // 网络输出标签对应类别名称初始化
    std::vector<std::string> comboBoxContents = datasetInfo->selectedClassNames;
    if(comboBoxContents.size()>0){
        for(int i=0;i<comboBoxContents.size();i++)   {label2class[i]=comboBoxContents[i];
            qDebug()<<"(MonitorPage::refresh) comboBoxContents[i]="<<QString::fromStdString(comboBoxContents[i]);}
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    std::map<std::string, int>::iterator iter;
    iter = class2label.begin();
    while(iter != class2label.end()) {
        std::cout << iter->first << " : " << iter->second << std::endl;
        iter++;
    }
    //如果数据集或模型路径变了
    if(
        modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH") != choicedModelPATH 
        ||
        datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH") != choicedDatasetPATH
    ){
        if(datasetInfo->selectedType=="INCRE") ifDataPreProcess=false;
        choicedModelPATH=modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
        choicedDatasetPATH=datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
        //TODO 因为使用测试页面，下面嗯切可能带来错误
        inferThread->setClass2LabelMap(class2label);
        //qDebug()<<"(MonitorPage::refresh) class2label.size()=="<<class2label.size();
        inferThread->setParmOfRTI(choicedModelPATH,ifDataPreProcess);//只有小样本是false 既不做预处理
        client->setClass2LabelMap(class2label);
        client->setParmOfRTI(choicedDatasetPATH);//发的数据不做归一化预处理
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
    for(int i=0;i<degrees.size();i++){
        degrees[i]=round(degrees[i] * 100) / 100;
    }
    QString predClass = QString::fromStdString(label2class[predIdx]);
    //terminal->print("Real-time classification results:"+predClass);//连续调用恐怕会有问题
    QWidget *tempWidget=tempChart->drawDisDegreeChart(predClass,degrees,label2class);
    removeLayout2(ui->horizontalLayout_degreeChart2);
    ui->horizontalLayout_degreeChart2->addWidget(tempWidget);
    ui->jcLabel->setText(QString::fromStdString(label2class[predIdx]));
}

void MonitorPage::enableSimulateSignal(){
    terminal->print("模型已载入可以开始模拟发送");
    ui->simulateSignal->setEnabled(true);
}

void MonitorPage::showColorMap(){
    /*=================draw thermal column==============*/
    QLabel *imageLabel=new QLabel;
    imageLabel->setBackgroundRole(QPalette::Base);
    imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel->setScaledContents(true);
    //imageLabel->setStyleSheet("border:2px solid red;");
    QImage image;
    QImageReader reader("./colorMap.png");
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
