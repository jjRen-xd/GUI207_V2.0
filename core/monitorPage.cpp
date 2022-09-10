#include "monitorPage.h"
#include "lib/algorithm/trtinfer.h"
#include <iostream>
#include <QMessageBox>
#include <QMutex>


MonitorPage::MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
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
    //connect(inferThread, &InferThread::sigInferResult,this,&MonitorPage::showInferResult);
    connect(inferThread, SIGNAL(sigInferResult(int,QVariant)),this,SLOT(showInferResult(int,QVariant)));
    inferThread->setInferMode("real_time_infer");

    server = new SocketServer(&sem,&sharedQue,&lock,terminal);//监听线程

    connect(ui->startListen, &QPushButton::clicked, this, &MonitorPage::StartListen);
    connect(ui->simulateSignal, &QPushButton::clicked, this, &MonitorPage::simulateSend);
    connect(ui->stopListen, &QPushButton::clicked,[this]() { delete server; });

}
void MonitorPage::StartListen(){
    if(modelInfo->selectedType==""){
        QMessageBox::warning(NULL, "实时监测", "监听失败，请先指定HRRP模型。");
        qDebug()<<"modelInfo->selectedType=="<<QString::fromStdString(modelInfo->selectedType);
        return;
    }

    inferThread->setParmOfRTI(modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH"),true);
    server->start();
    terminal->print("开始监听");
    inferThread->start();
}

void MonitorPage::simulateSend(){
    SocketClient* client = new SocketClient();
    client->start();
}

void MonitorPage::refresh(){
    if(modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH")!=this->choicedModelPATH){
        trtInfer = new TrtInfer(class2label);
        this->choicedModelPATH=modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
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
    terminal->print("Real-time classification results:"+predClass);
    QWidget *tempWidget=tempChart->drawDisDegreeChart(predClass,degrees,label2class);
    QWidget *tempWidget2=tempChart->drawDisDegreeChart(predClass,degrees,label2class);
    removeLayout2(ui->horizontalLayout_degreeChart2);
    ui->horizontalLayout_degreeChart2->addWidget(tempWidget);
    ui->horizontalLayout_HotCol->addWidget(tempWidget2);
}
MonitorPage::~MonitorPage(){

}
