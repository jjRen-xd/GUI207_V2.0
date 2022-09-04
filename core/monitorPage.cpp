#include "monitorPage.h"
#include "lib/guiLogic/tools/realtimeinferencebuffer.h"
#include <iostream>
#include <QMessageBox>

MonitorPage::MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    modelInfo(globalModelInfo)
{
    label2class[0] ="bigball"; //"XQ";'bigball','DT','Moxiu','sallball','taper', 'WD'
    label2class[1] ="DT"; //"DQ";
    label2class[2] ="Moxiu"; //"Z";
    label2class[3] ="sallball"; //"QDZ";
    label2class[4] ="taper"; //"DT";
    label2class[5] ="WD"; //"FG";
    for(auto &item: label2class){
        class2label[item.second] = item.first;
    }
    que = new RealTimeInferenceBuffer();
    server = new SocketServer(bash_terminal);
    connect(ui->startListen, &QPushButton::clicked, this, &MonitorPage::StartListen);
    connect(ui->stopListen, &QPushButton::clicked,[this](){
        delete server;
    });

}
void MonitorPage::StartListen(){
    if(modelInfo->selectedType!=""){
        QMessageBox::warning(NULL, "实时监测", "监听失败，请先指定HRRP模型。");
        qDebug()<<"modelInfo->selectedType=="<<QString::fromStdString(modelInfo->selectedType);
        return;
    }
    std::thread th(&SocketServer::Start,server,que);
    std::thread th_infer(&TrtInfer::realTimeInfer,trtInfer,que,modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH"), true);

    th_infer.detach();
    th.detach();

    terminal->print("开始监听");
}

void MonitorPage::refresh(){
    if(modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH")!=this->choicedModelPATH){
        trtInfer = new TrtInfer(ui,class2label);
        this->choicedModelPATH=modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
    }
}

MonitorPage::~MonitorPage(){

}
