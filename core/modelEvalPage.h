#ifndef MODELEVALPAGE_H
#define MODELEVALPAGE_H

#include <vector>
#include <string>
#include <map>
#include <QObject>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"

#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./core/datasetsWindow/chart.h"

#include "./lib/guiLogic/tools/searchFolder.h"

#include "lib/algorithm/libtorchTest.h"


class ModelEvalPage:public QObject{
    Q_OBJECT
public:
    ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo);
    ~ModelEvalPage();

    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;

    void disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames);

public slots:
    void refreshGlobalInfo();

    // 针对全部样本

    // 针对单样本
    void randSample();
    void testOneSample();
    void testAllSample();

private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    DatasetInfo *datasetInfo;
    ModelInfo *modelInfo;

    std::string choicedDatasetPATH;
    std::string choicedModelPATH;
    std::string choicedSamplePATH;

    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();

    LibtorchTest *libtorchTest;

};

#endif // MODELEVALPAGE_H
