#ifndef MONITORPAGE_H
#define MONITORPAGE_H

#include <QObject>
#include <thread>
#include "./core/datasetsWindow/chart.h"
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/tools/socketserver.h"
#include "lib/algorithm/inferthread.h"

#include "./lib/algorithm/trtinfer.h"


class MonitorPage : public QObject
{
    Q_OBJECT
public:
    MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo);
    void StartListen();
    SocketServer* server;
    InferThread* inferThread=nullptr;
    RealTimeInferenceBuffer* que;
    std::queue<std::vector<float>> sharedQue;
    TrtInfer* trtInfer;
    void refresh();
    ~MonitorPage();



public slots:
    void showInferResult(int,QVariant);

private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    std::string choicedModelPATH;
    ModelInfo *modelInfo;
    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;


};

#endif // MONITORPAGE_H
