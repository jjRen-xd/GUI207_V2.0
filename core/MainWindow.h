#pragma once
#include <QMainWindow>
#include <QProcess>
// 数据记录类
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/modelInfo.h"
// 主页面类
#include "./core/sensePage.h"
#include "./core/modelChoicePage.h"
#include "./core/modelEvalPage.h"
#include "./core/modelTrainPage.h"
#include "./core/monitorPage.h"

//#include "./lib/guiLogic/modelEval.h"
// 悬浮窗部件类
#include "./core/datasetsWindow/datasetDock.h"
#include "./core/modelsWindow/modelDock.h"
#include "./lib/guiLogic/bashTerminal.h"
// 界面美化类
#include "./conf/QRibbon/QRibbon.h"


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    static const std::string slash="\\";
#else
    static const std::string slash="/";
#endif

namespace Ui{
    class MainWindow; 
};

class MainWindow: public QMainWindow{
	Q_OBJECT

    public:
        MainWindow(QWidget *parent = Q_NULLPTR);
        ~MainWindow();

        BashTerminal *terminal; // 自定义终端
//        ModelEval *modeleval; // 模型评估页面控制类
    public slots:
        void switchPage();      // 页面切换
        void fullScreen();      // 全屏

    private:
        Ui::MainWindow *ui; 
        
        DatasetDock *datasetDock;
        ModelDock *modelDock;

        SenseSetPage *senseSetPage;
        ModelChoicePage *modelChoicePage;
        ModelEvalPage *modelEvalPage;
        ModelTrainPage *modelTrainPage;
        MonitorPage *monitorPage;

        DatasetInfo *globalDatasetInfo;
        ModelInfo *globalModelInfo;

};
