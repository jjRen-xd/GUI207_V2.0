#include "monitorPage.h"

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
    //que = new RealTimeInferenceBuffer();
    QSemaphore sem;
    QMutex lock;

    inferThread =new InferThread(&sem,&sharedQue,&lock);
    //connect(inferThread, &InferThread::sigInferResult,this,&MonitorPage::showInferResult);
    connect(inferThread, SIGNAL(sigInferResult(QString)),this,SLOT(showInferResult(QString)));

    inferThread->setInferMode("real_time_infer");
    server = new SocketServer(&sem,&sharedQue,&lock,terminal);

    connect(ui->startListen, &QPushButton::clicked, this, &MonitorPage::StartListen);
    connect(ui->stopListen, &QPushButton::clicked,[this](){
        delete server;
    });



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
void MonitorPage::showInferResult(QString s){
    qDebug()<<"(MonitorPage::showInferResult) heerererererererreeeeeeeeeeeeeeeeeeeee"<<s;
    ui->onetemplabel->setText(s);

    Chart *tempChart = new Chart(ui->label_mE_chartGT,"","");//就调用一下它的方法
    std::vector<float> degrees={0.1,0.1,0.1,0.1,0.2,0.4};
    QString asdf="DT";
    QWidget *tempWidget=tempChart->drawDisDegreeChart(asdf,degrees,label2class);
    removeLayout2(ui->horizontalLayout_degreeChart2);
    ui->horizontalLayout_degreeChart2->addWidget(tempWidget);

//    QChart *chart = new QChart;
//    //qDebug() << "(ModelEvalPage::disDegreeChart)子线程id：" << QThread::currentThreadId();
//    std::map<QString, std::vector<float>> mapnum;
//    mapnum.insert(std::pair<QString, std::vector<float>>(predClass_QString, degrees));  //后续可拓展
//    QBarSeries *series = new QBarSeries();
//    std::map<QString, std::vector<float>>::iterator it = mapnum.begin();

//    //将数据读入
//    while (it != mapnum.end()){
//        QString tit = it->first;
//        QBarSet *set = new QBarSet(tit);
//        std::vector<float> vecnum = it->second;
//        for (auto &a : vecnum){
//            *set << a;
//        }
//        series->append(set);
//        it++;
//    }
//    series->setVisible(true);
//    series->setLabelsVisible(true);
//    // 横坐标参数
//    QBarCategoryAxis *axis = new QBarCategoryAxis;
//    for(int i = 0; i<label2class.size(); i++){
//        axis->append(QString::fromStdString(label2class[i]));
//    }
//    QValueAxis *axisy = new QValueAxis;
//    axisy->setTitleText("隶属度");
//    chart->addSeries(series);
//    chart->setTitle("识别目标对各类别隶属度分析图");
//    //std::cout<<"(ModelEvalPage::disDegreeChart): H444444444444"<<std::endl;
//    chart->setAxisX(axis, series);
//    chart->setAxisY(axisy, series);
//    chart->legend()->setVisible(true);

//    QChartView *view = new QChartView(chart);
//    view->setRenderHint(QPainter::Antialiasing);
//    removeLayout(ui->horizontalLayout_degreeChart);
//    ui->horizontalLayout_degreeChart->addWidget(view);
//    QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");
}

MonitorPage::~MonitorPage(){

}
