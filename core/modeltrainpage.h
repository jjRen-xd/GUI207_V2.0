#ifndef MODELTRAINPAGE_H
#define MODELTRAINPAGE_H


#include <QObject>
#include <QMessageBox>
#include <QFileDialog>
#include <windows.h>
#include <mat.h>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include "./core/modelsWindow/modelDock.h"

class ModelTrainPage:public QObject
{
    Q_OBJECT
public:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    DatasetInfo *datasetInfo;
    ModelInfo *modelInfo;
    BashTerminal *train_terminal;
    ModelDock *modelDock;

    QString choicedDatasetPATH;
    QProcess *processTrain;
    std::vector<std::string> modelTypes={"HRRP","AFS","FewShot"};
    int trainModelType=0;
    QString cmd="";
    QString time = "";
    QString batchSize = "";
    QString epoch = "";
    QString saveModelName = "";

    ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,
                   ModelInfo *globalModelInfo, ModelDock *modelDock);
    void refreshGlobalInfo();
    void uiInitial();
    void execuCmd(QString cmd);   // 开放在终端运行命令接口
    void showTrianResult();
//    int getDataLen(std::string dataPath);
//    int getDataClassNum(std::string dataPath, std::string specialDir);

    // 为了兼容win与linux双平台
    bool showLog=false;
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    QString bashApi = "powershell";            // "Windows" or "Linux"
    #else
    QString bashApi = "bash";            // "Windows" or "Linux"
    #endif

public slots:
    void startTrain();
    void stopTrain();
    void monitorTrainProcess();
    void changeTrainType();
//    void editModelFile();
//    void chooseOldClass();

signals:

private:
//    QString dataDir;
//    int batchSize;
//    int maxEpoch;

};



#endif // MODELTRAINPAGE_H
