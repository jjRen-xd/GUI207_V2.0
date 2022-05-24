#ifndef SENSEPAGE_H
#define SENSEPAGE_H

#include <vector>
#include <map>
#include <QObject>
#include <QButtonGroup>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./core/datasetsWindow/chart.h"

#include "./lib/guiLogic/tools/searchFolder.h"

class SenseSetPage:public QObject{
    Q_OBJECT
public:
    SenseSetPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo);
    ~SenseSetPage();

    std::map<std::string, QLineEdit*> attriLabelGroup;
    std::vector<QLabel*> imgGroup;
    std::vector<QLabel*> imgInfoGroup;
    std::vector<QLabel*> chartGroup;
    std::vector<QLabel*> chartInfoGroup;
    QButtonGroup *BtnGroup_typeChoice = new QButtonGroup;



public slots:
    void changeType();
    void confirmDataset(bool notDialog);
    void saveDatasetAttri();

    void updateAttriLabel();
    void drawClassImage();
    void nextBatchChart();


private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;

    DatasetInfo *datasetInfo;

    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();

};

#endif // SENSEPAGE_H
