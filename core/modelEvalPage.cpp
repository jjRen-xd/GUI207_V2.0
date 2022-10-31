#include "modelEvalPage.h"
#include <QMessageBox>
#include <QGraphicsScene>
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
    label2class.clear();
    class2label.clear();
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
    if((modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH")!=choicedModelPATH)||
    (datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH")!=choicedDatasetPATH)){//保证模型切换后trt对象重新构建
        trtInfer = new TrtInfer(class2label);
        choicedDatasetPATH = datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
        choicedModelPATH=modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH");
    }
}


void ModelEvalPage::randSample(){
    // 获取下拉框类别内容
    string selectedClass = ui->comboBox_sampleType->currentText().toStdString();
    // 已选类别的随机取样
    if(!selectedClass.empty()){
        string classPath = choicedDatasetPATH +"/" +selectedClass;
        // string datafileFormat =datasetInfo->getAttri(datasetInfo->selectedType, datasetInfo->selectedName, "dataFileFormat");
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
    struct stat buffer; 
    int modelfileExist=(stat (choicedModelPATH.c_str(), &buffer) == 0);
    if(!choicedModelPATH.empty() && !choicedSamplePATH.empty()&& (modelfileExist==1)){
        std::cout<<"(ModelEvalPage::testOneSample)choicedSamplePATH"<<choicedSamplePATH<<endl;
        std::vector<float> degrees; int predIdx;
        //classnum==(datasetInfo->selectedClassNames.size())
        std::cout<<"(ModelEvalPage::testOneSample)datasetInfo->selectedType="<<datasetInfo->selectedType<<endl;
        std::cout<<"(ModelEvalPage::testOneSample)modelInfo->selectedType="<<modelInfo->selectedType<<endl;
        bool dataProcess=true;std::string flag="";
        if(modelInfo->selectedType=="INCRE") dataProcess=false; //目前的增量模型接受的数据是没做预处理的
        if(modelInfo->selectedType=="FEA_RELE"){
            std::string feaWeightTxtPath=choicedModelPATH.substr(0, choicedModelPATH.rfind("/"))+"/model/attention.txt";
            if(this->dirTools->exist(feaWeightTxtPath)){//判断是abfc还是atec,依据就是有没有attention文件
                int modelIdx=1,tempi=0;std::vector<int> dataOrder;std::string line;
                std::ifstream infile(feaWeightTxtPath);
                while (getline(infile, line)){
                    if(++tempi==40){modelIdx=std::stoi(line);break;}
                    dataOrder.push_back(std::stoi(line));
                }infile.close();
                trtInfer->setParmsOfAFS(modelIdx, dataOrder);
                flag="FEA_RELE_abfc";
            }
            else{
                flag="FEA_RELE_atec";
                dataProcess=false;
            }
        }
        trtInfer->testOneSample(choicedSamplePATH, this->emIndex, choicedModelPATH, dataProcess , &predIdx, degrees, flag);

        /*************************把下面都当做对UI的操作***************************/
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
//        ui->label_predTime->setText(QString("%1").arg(predTime));
        QString imgPath = QString::fromStdString(choicedDatasetPATH) +"/"+ predClass +".png";
        ui->label_predImg->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));

        // 绘制隶属度柱状图
        disDegreeChart(predClass, degrees, label2class);

    }
    else{
        QMessageBox::warning(NULL, "单样本测试", "数据或模型未指定！(检查模型路径是否存在)");
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
        std::vector<std::vector<int>> confusion_matrix(classNum, std::vector<int>(classNum, 0));
        //libtorchTest->testAllSample(choicedDatasetPATH, choicedModelPATH, acc, confusion_matrix);
        //onnxInfer->testAllSample(choicedDatasetPATH, choicedModelPATH, acc, confusion_matrix);

        bool dataProcess=true;
        std::string flag="";
        if(modelInfo->selectedType=="INCRE") dataProcess=false; //目前增量模型接受的数据是不做预处理的

        if(modelInfo->selectedType=="FEA_RELE"){
            std::string feaWeightTxtPath=choicedModelPATH.substr(0, choicedModelPATH.rfind("/"))+"/model/attention.txt";
            if(this->dirTools->exist(feaWeightTxtPath)){//判断是abfc还是atec,依据就是有没有attention文件
                int modelIdx=1,tempi=0;std::vector<int> dataOrder;std::string line;
                std::ifstream infile(feaWeightTxtPath);
                while (getline(infile, line)){
                    if(++tempi==40){modelIdx=std::stoi(line);break;}
                    dataOrder.push_back(std::stoi(line));
                    //cout<<std::stoi(line)<<endl;
                }infile.close();
                trtInfer->setParmsOfAFS(modelIdx, dataOrder);
                flag="FEA_RELE_abfc";
            }
            else{
                flag="FEA_RELE_atec";
                dataProcess=false;
            }
        }

        if(!trtInfer->testAllSample(choicedDatasetPATH,choicedModelPATH,inferBatch,dataProcess,acc,confusion_matrix,flag)){
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
        pRet = (PyArrayObject*)PyObject_CallObject(pFunc, args);
        delete [ ] numpyptr;
        qDebug()<<"(ModelEvalPage::testAllSample) python done";
        QString imgPath = QString::fromStdString("D:/confusion_matrix.jpg");

        if(all_Images[ui->graphicsView_3_evalpageMatrix]){ //delete 原来的图
            qgraphicsScene->removeItem(all_Images[ui->graphicsView_3_evalpageMatrix]);
            delete all_Images[ui->graphicsView_3_evalpageMatrix]; //空悬指针
            all_Images[ui->graphicsView_3_evalpageMatrix]=NULL;
        }
        if(this->dirTools->exist(imgPath.toStdString())){
            recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_3_evalpageMatrix);
        }
        //ui->label_evalpageMatrix->setPixmap(QPixmap(imgPath).scaled(QSize(576,432), Qt::KeepAspectRatio));
        //ui->label_evalpageMatrix->setPixmap(QPixmap(imgPath).scaled(ui->label_evalpageMatrix->size(), Qt::KeepAspectRatio));
        /*************************Draw******************************/
        QMessageBox::information(NULL, "所有样本测试", "识别成果，结果已输出！");
        ui->label_testAllAcc->setText(QString("%1").arg(acc*100));
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
    QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");
}

void ModelEvalPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    //QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image);  //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.95;              //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.95;           //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}