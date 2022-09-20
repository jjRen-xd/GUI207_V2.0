#ifndef SOCKETSERVER_H
#define SOCKETSERVER_H
#include <winsock.h>   // windows平台的网络库头文件
#include <queue>
#include <QThread>
#include <QDebug>
#include <QMutex>
#include <QMutexLocker>
#include <QSemaphore>
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#undef slots
#include <Python.h>
#include "arrayobject.h"
#define slots Q_SLOTS

class SocketServer:public QThread
{
    Q_OBJECT   //申明需要信号与槽机制支持
public:
    SocketServer(QSemaphore *sem,std::queue<std::vector<float>>* sharedQue,QMutex *l,BashTerminal *bash_terminal);
    ~SocketServer();
    void initialization();
    SOCKET createServeSocket(const char* ip);
    void run();

    //定义服务端套接字，接受请求套接字
    SOCKET s_server;
    SOCKET s_accept;
    QSemaphore *sem;
    std::queue<std::vector<float>>* sharedQue;
    QMutex *lock;
signals:
    void sigColorMap();

private:
    BashTerminal *terminal;
    void dataVisualization();
    std::deque<std::vector<float>> colorMapMatrix;
    PyObject *pModule,*pFunc,*PyArray,*args;
    PyArrayObject* pRet;
};

#endif // SOCKETSERVER_H
