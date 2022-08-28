#ifndef MODELTRAIN_H
#define MODELTRAIN_H


#include "qdatetime.h"
#include "ui_MainWindow.h"
#include <QObject>
#include <QProcess>
#include <QTextBrowser>

class ModelTrain: public QObject{
    Q_OBJECT

    public:
        ModelTrain(QTextBrowser* Widget, QLabel* trainImg, QLabel* valImg, QLabel* confusionMat,
                   QTextEdit* timeRestEdit, QProgressBar* trainProgressBar);
        void startTrain(QString cmd);   // 开放在终端运行命令接口
        void stopTrain();
        ~ModelTrain();

        QString dataRoot;
        int modelTypeId;
        bool showLog=true;

    private:
        QTextBrowser *OutShow;
        QLabel* trainImg;
        QLabel* valImg;
        QLabel* confusionMat;
        QTextEdit* timeRestEdit;
        QProgressBar* trainProgressBar;
        QProcess *process_train;    // 终端命令行输出

        // 为了兼容win与linux双平台
        #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
        QString bashApi = "powershell";            // "Windows" or "Linux"
        #else
        QString bashApi = "bash";            // "Windows" or "Linux"
        #endif

    private slots:
        void readLogOutput();      // 读取终端输出并显示
        void readLogError();       // 读取终端Error并显示
        void showTrianResult();
};











#endif // MODELTRAIN_H
