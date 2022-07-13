#include "modelEvalPage.h"
#include <QMessageBox>

#include <QChart>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>


using namespace std;

ModelEvalPage::ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo)
{
    // 网络输出标签对应类别名称初始化
    label2class[0] = "XQ";
    label2class[1] = "DQ";
    label2class[2] = "Z";
    label2class[3] = "QDZ";
    label2class[4] = "DT";
    for(auto &item: label2class){
        class2label[item.second] = item.first;
    }

    // 先用libtorch
    libtorchTest = new LibtorchTest(class2label);
    //
    //onnxInfer = new OnnxInfer(class2label);
    // 随机选取样本按钮
    connect(ui->pushButton_mE_randone, &QPushButton::clicked, this, &ModelEvalPage::randSample);
    // 测试按钮
    connect(ui->pushButton_testOneSample, &QPushButton::clicked, this, &ModelEvalPage::testOneSample);
    connect(ui->pushButton_testAllSample, &QPushButton::clicked, this, &ModelEvalPage::testAllSample);
}

ModelEvalPage::~ModelEvalPage(){

}


void ModelEvalPage::refreshGlobalInfo(){
    // 基本信息更新
    ui->label_mE_dataset->setText(QString::fromStdString(datasetInfo->selectedName));
    ui->label_mE_model->setText(QString::fromStdString(modelInfo->selectedName));
    ui->label_mE_batch->setText(QString::fromStdString(modelInfo->getAttri(modelInfo->selectedType, modelInfo->selectedName, "batch")));
    this->choicedDatasetPATH = datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
    this->choicedModelPATH = modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
    // 单样本测试下拉框刷新
    vector<string> comboBoxContents = datasetInfo->selectedClassNames;
    ui->comboBox_sampleType->clear();
    for(auto &item: comboBoxContents){
        ui->comboBox_sampleType->addItem(QString::fromStdString(item));
    }

}


void ModelEvalPage::randSample(){
    // 获取下拉框类别内容
    string selectedClass = ui->comboBox_sampleType->currentText().toStdString();
    // 已选类别的随机取样
    if(!selectedClass.empty()){
        string classPath = choicedDatasetPATH +"/" +selectedClass;
        vector<string> sampleNames;

        if(dirTools->getFiles(sampleNames,".txt",classPath)){
            srand((unsigned)time(NULL));

            string choicedFile = sampleNames[(rand())%sampleNames.size()];
            QString txtFilePath = QString::fromStdString(classPath + "/" + choicedFile);
            this->choicedSamplePATH = txtFilePath.toStdString();

            // 可视化所选样本
            ui->label_mE_choicedSample->setText(QString::fromStdString(choicedFile).split(".").first());

            QString imgPath = QString::fromStdString(choicedDatasetPATH +"/"+ selectedClass +".png");
            ui->label_mE_imgGT->setPixmap(QPixmap(imgPath).scaled(QSize(100,100), Qt::KeepAspectRatio));

            Chart *previewChart = new Chart(ui->label_mE_chartGT,"HRRP(Ephi),Polarization HP(1)[Magnitude in dB]",txtFilePath);
            previewChart->drawHRRPimage(ui->label_mE_chartGT);
        }
    }
    else{
        QMessageBox::warning(NULL, "数据取样", "数据取样失败，请指定数据集类型!");
    }


}


void ModelEvalPage::testOneSample(){
    if(!choicedModelPATH.empty() && !choicedSamplePATH.empty()){
        std::cout<<choicedSamplePATH<<endl;
        std::vector<float> degrees(datasetInfo->selectedClassNames.size());  //隶属度
        //int predIdx = libtorchTest->testOneSample(choicedSamplePATH, choicedModelPATH, degrees);
        int predIdx = onnxInfer->testOneSample(choicedSamplePATH, choicedModelPATH, degrees);
        QString predClass = QString::fromStdString(label2class[predIdx]);   // 预测类别

        terminal->print("识别结果： " + predClass);
        terminal->print(QString("隶属度：%1").arg(degrees[predIdx]));

        // 可视化结果
        ui->label_predClass->setText(predClass);
        ui->label_predDegree->setText(QString("%1").arg(degrees[predIdx]*100));
        QString imgPath = QString::fromStdString(choicedDatasetPATH) +"/"+ predClass +".png";
        ui->label_predImg->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));

        // 绘制隶属度柱状图
        disDegreeChart(predClass, degrees, label2class);

        QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");
    }
    else{
        QMessageBox::warning(NULL, "单样本测试", "数据或模型未指定！");
    }
}



// 移除布局子控件
void removeLayout(QLayout *layout){
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
            removeLayout(childLayout);

        delete child;
        child = nullptr;
    }
}


void ModelEvalPage::disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames){
    QChart *chart = new QChart;
//    std::vector<int> list0 = { 101,505,200,301 };

    std::map<QString, vector<float>> mapnum;
    mapnum.insert(pair<QString, vector<float>>(classGT, degrees));  //后续可拓展

    QBarSeries *series = new QBarSeries();
    map<QString, vector<float>>::iterator it = mapnum.begin();
    //将数据读入
    while (it != mapnum.end()){
        QString tit = it->first;
        QBarSet *set = new QBarSet(tit);
        std::vector<float> vecnum = it->second;
        for (auto &a : vecnum){
            *set << a;
        }
        series->append(set);
        it++;
    }
    series->setVisible(true);
    series->setLabelsVisible(true);
    // 横坐标参数
    QBarCategoryAxis *axis = new QBarCategoryAxis;
    for(int i = 0; i<classNames.size(); i++){
        axis->append(QString::fromStdString(classNames[i]));
    }
    QValueAxis *axisy = new QValueAxis;
    axisy->setTitleText("隶属度");
    chart->addSeries(series);
    chart->setTitle("识别目标对各类别隶属度分析图");

    chart->setAxisX(axis, series);
    chart->setAxisY(axisy, series);
    chart->legend()->setVisible(true);

    QChartView *view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    removeLayout(ui->horizontalLayout_degreeChart);
    ui->horizontalLayout_degreeChart->addWidget(view);
}



// TODO 待优化
void ModelEvalPage::testAllSample(){
    if(!choicedDatasetPATH.empty() && !choicedModelPATH.empty()){
        float acc = 0.0;
        std::vector<std::vector<int>> confusion_matrix(5, std::vector<int>(5, 0));
        libtorchTest->testAllSample(choicedDatasetPATH, choicedModelPATH, acc, confusion_matrix);
        QMessageBox::information(NULL, "所有样本测试", "识别成果，结果已输出！");

        ui->label_testAllAcc->setText(QString("%1").arg(acc*100));
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                QLabel *valuelabel = ui->confusion_matrix->findChild<QLabel *>("cfmx_"+QString::number(i)+QString::number(j));
                valuelabel->setText(QString::number(confusion_matrix[i][j]));
            }
        }
    }
    else{
        QMessageBox::warning(NULL, "所有样本测试", "数据集或模型未指定！");
    }
}

