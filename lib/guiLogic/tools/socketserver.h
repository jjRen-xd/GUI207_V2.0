#ifndef SOCKETSERVER_H
#define SOCKETSERVER_H
#include <winsock.h>   // windows平台的网络库头文件
#include "lib/guiLogic/tools/realtimeinferencebuffer.h"
#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"

class SocketServer
{
public:
    SocketServer(BashTerminal *bash_terminal);

    void initialization();
    SOCKET createServeSocket(const char* ip);
    void Start(RealTimeInferenceBuffer* que);


    //定义服务端套接字，接受请求套接字
    SOCKET s_server;
    SOCKET s_accept;
private:
    BashTerminal *terminal;
};

#endif // SOCKETSERVER_H
