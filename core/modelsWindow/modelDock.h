#ifndef MODELDOCK_H
#define MODELDOCK_H

#include <QObject>
#include "ui_MainWindow.h"

#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"

#include "./lib/guiLogic/tools/searchFolder.h"


class ModelDock:public QObject{
    Q_OBJECT
public:
    ModelDock(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo);
    ~ModelDock(){};

    std::map<std::string, QLabel*> attriLabelGroup;

    void reloadTreeView();

    std::string previewType;
    std::string previewName;

 public slots:
     void importModel(std::string type);
     void importModelAfterTrain(std::string type, QString modelName);
     void deleteModel();


private slots:
     void treeItemClicked(const QModelIndex &index);
private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;

    ModelInfo *modelInfo;

    QTreeView *modelTreeView;

    // 不同平台下文件夹搜索工具
    SearchFolder *dirTools = new SearchFolder();
};

#endif // MODELDOCK_H
