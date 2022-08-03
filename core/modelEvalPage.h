#ifndef MODELEVALPAGE_H
#define MODELEVALPAGE_H

#include <vector>
#include <string>
#include <map>
#include <QObject>
#include <QThread>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"

#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./core/datasetsWindow/chart.h"

#include "./lib/guiLogic/tools/searchFolder.h"

#include "lib/algorithm/libtorchTest.h"
#include "lib/algorithm/onnxinfer.h"
#include "lib/algorithm/trtinfer.h"

class ModelEvalPage:public QObject{
    Q_OBJECT
public:
    ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo);
    ~ModelEvalPage();

    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;

    void disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames);
    void testOneSample_ui();
    friend void testOneSample_ui2(ModelEvalPage dv);
    //推理需要的全局变量
    std::promise<int> predIdx_promise;
    std::promise<std::vector<float>> degrees_promise;
    std::future<int> predIdx_future;
    std::future<std::vector<float>> degrees_future;

    Ui_MainWindow *ui;
    BashTerminal *terminal;


public slots:
    void refreshGlobalInfo();

    // 针对全部样本

    // 针对单样本
    void randSample();
    void testOneSample();

    void testAllSample();
signals:
    void stating(std::string choicedsamplePATH,std::string choicedmodelPATH,std::vector<float> &degrees);
private:
//    Ui_MainWindow *ui;
//    BashTerminal *terminal;
    DatasetInfo *datasetInfo;
    ModelInfo *modelInfo;

//    std::string choicedDatasetPATH="E:/207Project/Data/HRRP";
//    std::string choicedModelPATH="E:/207Project/GUI207_V2.0/db/models/onnxmodel.trt";
//    std::string choicedSamplePATH="E:/207Project/Data/HRRP/DT/00.txt";
    std::string choicedDatasetPATH;
    std::string choicedModelPATH;
    std::string choicedSamplePATH;
    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();

    //推理算法
    LibtorchTest *libtorchTest;
    OnnxInfer *onnxInfer;
    TrtInfer *trtInfer;
    QThread *qthread1;
};

#endif // MODELEVALPAGE_H
