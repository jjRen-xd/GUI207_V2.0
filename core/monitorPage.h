#ifndef MONITORPAGE_H
#define MONITORPAGE_H

#include <QObject>
#include <thread>
#include "./core/datasetsWindow/chart.h"
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"
#include "./lib/guiLogic/datasetInfo.h"
#include "./lib/guiLogic/tools/socketserver.h"
#include "./lib/guiLogic/tools/socketclient.h"
#include "lib/algorithm/inferthread.h"

class MonitorPage : public QObject
{
    Q_OBJECT
public:
    MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,ModelInfo *globalModelInfo);
    void startListen();
    void stopSend();
    void simulateSend();
    SocketServer* server;
    InferThread* inferThread=nullptr;
    SocketClient* client;
    void paintLabel();
    std::queue<std::vector<float>> sharedQue;
    TrtInfer* trtInfer;
    void refresh();
    //bool eventFilter(QObject *watched, QEvent *event);
    ~MonitorPage();



public 
slots:
    void showInferResult(int,QVariant);
    void enableSimulateSignal();
    void showColorMap();
    void showRealClass(int);
signals:
    void startOrstop_sig(bool);

private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;
    std::string choicedDatasetPATH="";
    std::string choicedModelPATH="";
    ModelInfo *modelInfo;
    DatasetInfo *datasetInfo;
    std::map<int, std::string> label2class;
    std::map<std::string, int> class2label;


};

#endif // MONITORPAGE_H
