#include "modelEvalPage.h"
#include <QMessageBox>

#include <QChart>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <thread>
#include "./lib/guiLogic/tools/guithreadrun.h"
#include<cuda_runtime.h>

#include<Windows.h>  //for Sleep func
using namespace std;

ModelEvalPage::ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo)
{
    GuiThreadRun::inst();
    // 随机选取样本按钮
    connect(ui->pushButton_mE_randone, &QPushButton::clicked, this, &ModelEvalPage::randSample);
    // 测试按钮
    connect(ui->pushButton_testOneSample, &QPushButton::clicked, this, &ModelEvalPage::testOneSample);
    connect(ui->pushButton_testAllSample, &QPushButton::clicked, this, &ModelEvalPage::testAllSample);

    Py_SetPythonHome(L"D:/win_anaconda");
    Py_Initialize();
    _import_array();
    PyRun_SimpleString("import sys");
    //PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('../../lib/guiLogic/tools/')");
    pModule = PyImport_ImportModule("EvalPageConfusionMatrix");
    pFunc = PyObject_GetAttrString(pModule, "draw_confusion_matrix");

}

ModelEvalPage::~ModelEvalPage(){

}

void ModelEvalPage::refreshGlobalInfo(){
    // 单样本测试下拉框刷新
    vector<string> comboBoxContents = datasetInfo->selectedClassNames;
    ui->comboBox_sampleType->clear();
    for(auto &item: comboBoxContents){
        ui->comboBox_sampleType->addItem(QString::fromStdString(item));
    }
    ui->comboBox_inferBatchsize->clear();
    for(int i=512;i>3;i/=2){
        ui->comboBox_inferBatchsize->addItem(QString::number(i));
    }
    // 网络输出标签对应类别名称初始化
    if(comboBoxContents.size()>0){
        for(int i=0;i<comboBoxContents.size();i++)   label2class[i]=comboBoxContents[i];
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    // 基本信息更新
    ui->label_mE_dataset->setText(QString::fromStdString(datasetInfo->selectedName));
    ui->label_mE_model->setText(QString::fromStdString(modelInfo->selectedName));
    //ui->label_mE_batch->setText(QString::fromStdString(modelInfo->getAttri(modelInfo->selectedType, modelInfo->selectedName, "batch")));
    this->choicedDatasetPATH = datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
    if(modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH")!=this->choicedModelPATH){//保证模型切换后trt对象重新构建
        trtInfer = new TrtInfer(class2label);
        this->choicedModelPATH=modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
    }
}


void ModelEvalPage::randSample(){
    // 获取下拉框类别内容
    string selectedClass = ui->comboBox_sampleType->currentText().toStdString();
    // 已选类别的随机取样
    if(!selectedClass.empty()){
        string classPath = choicedDatasetPATH +"/" +selectedClass;
        string datafileFormat =datasetInfo->getAttri(datasetInfo->selectedType, datasetInfo->selectedName, "dataFileFormat");
        srand((unsigned)time(NULL));
        Chart *previewChart;

        vector<string> allMatFile;
        if(dirTools->getFiles(allMatFile, ".mat", classPath)){
            QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
            this->choicedSamplePATH = matFilePath.toStdString();

            QString imgPath = QString::fromStdString(choicedDatasetPATH +"/"+ selectedClass +".png");
            //下面这部分代码都是为了让randomIdx在合理的范围内（
            MATFile* pMatFile = NULL;
            mxArray* pMxArray = NULL;
            pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
            if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空！！！！！！";return;}
            std::string matVariable=allMatFile[0].substr(0,allMatFile[0].find_last_of('.')).c_str();//假设数据变量名同文件名的话

            QString chartTitle="Temporary Title";
            if(datasetInfo->selectedType=="HRRP") {chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";}// matVariable="hrrp128";}
            else if (datasetInfo->selectedType=="RADIO") {chartTitle="RADIO Temporary Title";}// matVariable="radio101";}

            pMxArray = matGetVariable(pMatFile,matVariable.c_str());
            if(!pMxArray){qDebug()<<"(ModelEvalPage::randSample)pMxArray变量没找到！！！！！！";return;}
            int N = mxGetN(pMxArray);  //N 列数
            int randomIdx = N-(rand())%N;

            this->emIndex=randomIdx;
            // 可视化所选样本
            ui->label_mE_choicedSample->setText("Index:"+QString::number(randomIdx));
            ui->label_mE_imgGT->setPixmap(QPixmap(imgPath).scaled(QSize(100,100), Qt::KeepAspectRatio));
            //绘图
            previewChart = new Chart(ui->label_mE_chartGT,chartTitle,matFilePath);
            previewChart->drawImage(ui->label_mE_chartGT,datasetInfo->selectedType,randomIdx);
        }
    }
    else{
    QMessageBox::warning(NULL, "数据取样", "数据取样失败，请指定数据集类型!");
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


void  ModelEvalPage::testOneSample(){
/*
    因为infer是在另一个类中实现的，所以infer后对UI的操作仍然放在这个EvalPage类中实现，不然还要传ui。
    但是UI的操作还是不能在主线程中进行，因为那还得等infer出来才能操作UI。所以另开一个线程，把
*/
    if(!choicedModelPATH.empty() && !choicedSamplePATH.empty()){
        std::cout<<"(ModelEvalPage::testOneSample)choicedSamplePATH"<<choicedSamplePATH<<endl;
        std::vector<float> degrees; int predIdx;
        //classnum==(datasetInfo->selectedClassNames.size())
        std::cout<<"(ModelEvalPage::testOneSample)datasetInfo->selectedType="<<datasetInfo->selectedType<<endl;
        std::cout<<"(ModelEvalPage::testOneSample)modelInfo->selectedType="<<modelInfo->selectedType<<endl;
        bool dataProcess=true;
        if(modelInfo->selectedType=="INCRE") dataProcess=false; //目前的增量模型接受的数据是没做预处理的
        trtInfer->testOneSample(choicedSamplePATH, this->emIndex, choicedModelPATH, dataProcess , &predIdx, degrees);

/////////////////把下面都当做对UI的操作
        std::cout<<"(ModelEvalPage::testOneSample)degrees:";
        for(int i=0;i<degrees.size();i++){
            std::cout<<degrees[i]<<" ";
        }
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
//        Chart *forDegreeChart = new Chart(ui->label_mE_chartGT,"","");//这里传的参没啥用，只是想在下面调用一下它的方法
//        removeLayout(ui->horizontalLayout_degreeChart2);
//        QWidget* view=forDegreeChart->drawDisDegreeChart(predClass, degrees, label2class);
//        ui->horizontalLayout_degreeChart2->addWidget(view);
//        QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");

    }
    else{
        QMessageBox::warning(NULL, "单样本测试", "数据或模型未指定！");
    }
}


// TODO 待优化
void ModelEvalPage::testAllSample(){
    // 获取批处理量
    int inferBatch = ui->comboBox_inferBatchsize->currentText().toInt();
    qDebug()<<"(ModelEvalPage::testAllSample)batchsize=="<<inferBatch;
    /*这里涉及到的全局变量有除了模型数据集路径，还有准确度和混淆矩阵*/
    if(!choicedDatasetPATH.empty() && !choicedModelPATH.empty() ){
        float acc = 0.6;
        int classNum=label2class.size();
        std::vector<std::vector<int>> confusion_matrix(classNum, std::vector<int>(classNum, 6));
        //libtorchTest->testAllSample(choicedDatasetPATH, choicedModelPATH, acc, confusion_matrix);
        //onnxInfer->testAllSample(choicedDatasetPATH, choicedModelPATH, acc, confusion_matrix);

        bool dataProcess=true;
        if(modelInfo->selectedType=="INCRE") dataProcess=false; //目前的增量模型接受的数据是没做预处理的
        if(!trtInfer->testAllSample(choicedDatasetPATH,choicedModelPATH, inferBatch, dataProcess, acc, confusion_matrix)){
            return ;
        }

        /*************************Draw******************************/
        int* numpyptr= new int[classNum*classNum];
        for(int i=0;i<classNum;i++){
            for(int j=0;j<classNum;j++){
                numpyptr[i*classNum+j]=confusion_matrix[i][j];
            }
        }
        npy_intp dims[2] = {classNum,classNum};//矩阵维度
        PyArray = PyArray_SimpleNewFromData(2, dims, NPY_INT, numpyptr);//将数据变为numpy
        //用tuple装起来传入python
        args = PyTuple_New(2);
        std::string stringparm="";
        for(int i=0;i<classNum;i++) stringparm=stringparm+label2class[i]+"#";
        PyTuple_SetItem(args, 0, Py_BuildValue("s", stringparm.c_str()));
        PyTuple_SetItem(args, 1, PyArray);
        //函数调用
        pRet = (PyArrayObject*)PyEval_CallObject(pFunc, args);
        delete [ ] numpyptr;
        qDebug()<<"(ModelEvalPage::testAllSample) python done";
        /*************************Draw******************************/
        QMessageBox::information(NULL, "所有样本测试", "识别成果，结果已输出！");
        ui->label_testAllAcc->setText(QString("%1").arg(acc*100));
        for(int i=0;i<6;i++){
            for(int j=0;j<6;j++){
                QLabel *valuelabel = ui->confusion_matrix->findChild<QLabel *>("cfmx_"+QString::number(i)+QString::number(j));
                valuelabel->setText(QString::number(confusion_matrix[i][j]));
            }
        }
    }
    else{
        QMessageBox::warning(NULL, "所有样本测试", "数据集或模型未指定！");
    }
}

void ModelEvalPage::disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames){
    QChart *chart = new QChart;
    //qDebug() << "(ModelEvalPage::disDegreeChart)子线程id：" << QThread::currentThreadId();
    std::map<QString, vector<float>> mapnum;
    mapnum.insert(pair<QString, vector<float>>(classGT, degrees));  //后续可拓展
    QBarSeries *series = new QBarSeries();
    map<QString, vector<float>>::iterator it = mapnum.begin();
    //std::cout<<"(ModelEvalPage::disDegreeChart): H22222222222"<<std::endl;
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
    //std::cout<<"(ModelEvalPage::disDegreeChart): H444444444444"<<std::endl;
    chart->setAxisX(axis, series);
    chart->setAxisY(axisy, series);
    chart->legend()->setVisible(true);

    QChartView *view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    removeLayout(ui->horizontalLayout_degreeChart);
    ui->horizontalLayout_degreeChart->addWidget(view);
    QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");
}

