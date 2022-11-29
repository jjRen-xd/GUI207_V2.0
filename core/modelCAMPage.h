#ifndef MODELCAMPAGE_H
#define MODELCAMPAGE_H

#include <QObject>
#include <QString>
#include <QGraphicsView>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "ui_MainWindow.h"

#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/tools/searchFolder.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/customWidget/imagewidget.h"


class ModelCAMPage:public QObject{
    Q_OBJECT
public:
    ModelCAMPage(
        Ui_MainWindow *main_ui, 
        BashTerminal *bash_terminal, 
        DatasetInfo *globalDatasetInfo, 
        ModelInfo *globalModelInfo
    );
    ~ModelCAMPage();

    // 从xml加载5级模型结构的暴力方法，不优雅 // TODO
    void loadModelStruct_L1(QStringList &currLayers);
    void loadModelStruct_L2(QStringList &currLayers);
    void loadModelStruct_L3(QStringList &currLayers);
    void loadModelStruct_L4(QStringList &currLayers);
    void loadModelStruct_L5(QStringList &currLayers);


public slots:
    // 页面切换初始化
    void refreshGlobalInfo();

    void refreshVisInfo();  // 刷新预览图像与可视化目标层
    void clearComboBox();   // 清空下拉框

    int randomImage();      // 随机从所选中的数据集中抽取图像
    int importImage();      // 手动导入图像

    void confirmVis();      // 可视化确认按钮事件
    void execuCmdProcess(QString cmd);
    void processVisFinished();   // 可视化脚本执行结束事件 


    // 5级下拉框相关槽接口，过于暴力，不优雅 // TODO
    void on_comboBox_L1(QString choicedLayer);
    void on_comboBox_L2(QString choicedLayer);
    void on_comboBox_L3(QString choicedLayer);
    void on_comboBox_L4(QString choicedLayer);
    void on_comboBox_L5(QString choicedLayer);

    void showCamFig(QString method);


private:
    Ui_MainWindow *ui;

    BashTerminal *terminal;
    DatasetInfo *datasetInfo;
    ModelInfo *modelInfo;

    SearchFolder *dirTools = new SearchFolder();

    // 选择的数据集、模型、样本信息
    std::string choicedDatasetPATH;
    std::string choicedModelPATH;
    QString choicedModelSuffix;
    QString choicedCamMethod;

    QString choicedSamplePATH;
    int choicedMatIdx;

    // 选择模型结构的xml文件、预览图像路径 // FIXME 后期需要结合系统
    std::string modelStructXmlPath;
    QString modelStructImgPath;
    QString modelCheckpointPath;

    QString camImgsSavePath;
    QString condaPath;
    QString condaEnvName;
    QString pythonApiPath;


    std::map<std::string, std::string> choicedLayer;

    // 可视化的目标层
    QString targetVisLayer;

    // 缩放图像组件
    std::map<QGraphicsView*, ImageWidget*> all_Images;     // 防止内存泄露
    void recvShowPicSignal(QPixmap image, QGraphicsView* graphicsView);

    // 可视化进程
    QProcess *processVis;

};


#endif // MODELCAMPAGE_H
