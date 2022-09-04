#ifndef MONITORPAGE_H
#define MONITORPAGE_H

#include <QObject>
#include <thread>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/tools/socketserver.h"
#include "./lib/algorithm/trtinfer.h"


class MonitorPage : public QObject
{
    Q_OBJECT
public:
    MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo);
    void StartListen();
    SocketServer* server;
    RealTimeInferenceBuffer* que;
    TrtInfer* trtInfer;
    void refresh();
    ~MonitorPage();
signals:


private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    std::string choicedModelPATH;
    ModelInfo *modelInfo;
    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;
};

#endif // MONITORPAGE_H
