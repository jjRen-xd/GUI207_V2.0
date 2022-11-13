/*
接收数据，可视化数据，是inferThread线程的生产者
*/
#include "socketserver.h"

#pragma comment(lib,"ws2_32.lib")   // 库文件
#define PORT 2287
#define RECEIVE_BUF_SIZ 512
#define ColorMapColumnNUM 128

SocketServer::SocketServer(QSemaphore *s,std::queue<std::vector<float>>* sharedQ,QMutex *l,BashTerminal *bash_terminal):
    sem(s),sharedQue(sharedQ),lock(l),terminal(bash_terminal)
{
    for(int i=0;i<ColorMapColumnNUM;i++){
        std::vector<float> temp(128,-1);
        colorMapMatrix.push_back(temp);
    }
    Py_SetPythonHome(L"E:/anacoda/anaconda");
    Py_Initialize();
    _import_array();
    PyRun_SimpleString("import sys");
    //PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('../../lib/guiLogic/tools/')");
    pModule = PyImport_ImportModule("hrrp2colormap");
    pFunc = PyObject_GetAttrString(pModule, "ff");

}
void SocketServer::initialization() {
    //初始化套接字库
    // WSA  windows socket async  windows异步套接字     WSAStartup启动套接字
    // parm1:请求的socket版本 2.2 2.1 1.0     parm2:传出参数    参数形式：WORD  WSADATA
    WORD w_req = MAKEWORD(2, 2);//版本号
    WSADATA wsadata;
    // 成功：WSAStartup函数返回零
    if (WSAStartup(w_req, &wsadata) != 0) {qDebug() << "初始化套接字库失败！" ;}
    else {qDebug() << "初始化套接字库成功！" ;}
}

SOCKET SocketServer::createServeSocket(const char* ip){
    //1.创建空的Socket
        //parm1:af 地址协议族 ipv4 ipv6
        //parm2:type 传输协议类型 流式套接字(SOCK_STREAM) 数据报
        //parm3：protocl 使用具体的某个传输协议
    SOCKET s_server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s_server == INVALID_SOCKET){
        qDebug() << "套接字创建失败！" ;
        WSACleanup();
    }
    else {//qDebug() << "套接字创建成功！" ;
    }
    //2.给socket绑定ip地址和端口号
    struct sockaddr_in server_addr;   // sockaddr_in, sockaddr  老版本和新版的区别
    server_addr.sin_family = AF_INET;  // 和创建socket时必须一样
    server_addr.sin_port = htons(PORT);       // 端口号  大端（高位）存储(本地)和小端（低位）存储(网络），两个存储顺序是反着的  htons 将本地字节序转为网络字节序
    server_addr.sin_addr.S_un.S_addr = inet_addr(ip); //inet_addr将点分十进制的ip地址转为二进制
    if (::bind(s_server, (SOCKADDR*)&server_addr, sizeof(SOCKADDR)) == -1) {
        qDebug() << "套接字绑定失败！" ;
        WSACleanup();
    }
    else {//qDebug() << "套接字绑定成功！" ;
    }

    //3.设置套接字为监听状态  SOMAXCONN 监听的端口数 右键转到定义为5
    if (listen(s_server, SOMAXCONN) < 0) {
        qDebug() << "设置监听状态失败！" ;
        WSACleanup();
    }
    else {qDebug() << "socket 初始化、创建、绑定、设置监听状态成功！" ;

    }
    return s_server;
}

void SocketServer::run(){           //Producer
    //qDebug()<<"start temp=="<<temp;
    //定义发送缓冲区和接受缓冲区长度
    char send_buf[5]={'o','k','a','y'};
    char recv_buf[RECEIVE_BUF_SIZ];

    initialization(); // 初始化启动套接字
    s_server = createServeSocket("127.0.0.1");
    qDebug() << "waiting client connect..." ;
    // 如果有客户端请求连接
    s_accept = accept(s_server, NULL, NULL);
    if (s_accept == INVALID_SOCKET) {
        qDebug() << "连接失败！" ;
        WSACleanup();
        return ;
    }
    // 可以和客户端进行通信了
    std::vector<float> dataFrame;//里面放大小为模型输入数据长度个浮点数，用以送进网络。
    std::vector<float> repeatDataFrame;
    while (!isInterruptionRequested()) {
        // recv从指定的socket接受消息
        int getCharNum=recv(s_accept, recv_buf, RECEIVE_BUF_SIZ, 0);
        if ( getCharNum > 0){
            std::string temp=recv_buf;//针对于接受的数据只有一个浮点数 TODO
            float num_float;
            QT_TRY{
                num_float= std::stof(temp);//针对于接受的数据只有 "一个" "浮点数"TODO
            }QT_CATCH(...){
                qDebug()<<"(SocketServer::run)data errrr";continue;
            }
            //qDebug() << "客户端信息:" << QString::number(num_float) ;
            //terminal->print("Receive:"+QString::number(num_float));
            if(dataFrame.size()==(128-1)){//之后要和选择的模型匹配起来！！TODO
                //while(sharedQue->size()>0){}
                dataFrame.push_back(num_float);
                for(int i=0;i<128;i++){
                    for(int j=0;j<64;j++)
                        repeatDataFrame.push_back(dataFrame[i]);
                }
                //QMutexLocker x(lock);//智能锁,在栈区使用结束会自动释放
                sharedQue->push(repeatDataFrame);
                sem->release(1);
                //qDebug()<<"(SocketServer::run) sem->release()"<<QString::number(num_float);
                //terminal->print("One Sample was received ");
                //colorMapMatrix更新
                colorMapMatrix.pop_back();
                colorMapMatrix.push_front(dataFrame);
                dataVisualization();
                dataFrame.clear();
                repeatDataFrame.clear();
            }
            else{
                dataFrame.push_back(num_float);
                //qDebug()<<"接收到第"<<dataFrame.size()<<"个";
            }
        }
        else {qDebug() << "接收失败！" ;break;}
        if (send(s_accept, send_buf, 5, 0) < 0) {qDebug() << "发送失败！" ;break;}
    }
    //关闭套接字
    closesocket(s_server);
    closesocket(s_accept);
    //释放DLL资源
    WSACleanup();
    return;
}

void SocketServer::dataVisualization(){
    //数据准备
    float numpyptr[128*ColorMapColumnNUM];
    std::deque<std::vector<float>> colorMapMatrix_copy=colorMapMatrix;
    //qDebug()<<"colorMapMatrix.size()="<<colorMapMatrix.size();
    //qDebug()<<"colorMapMatrix_copy.size()="<<colorMapMatrix_copy.size();
    for(int i=0;i<ColorMapColumnNUM;i++){
        std::vector<float> asdf=colorMapMatrix_copy.front();
        for(int j=0;j<128;j++){
            numpyptr[i*128+j]=asdf[j];
        }
        colorMapMatrix_copy.pop_front();
    }
    npy_intp dims[2] = {ColorMapColumnNUM,128};//矩阵维度
    PyArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, numpyptr);//将数据变为numpy
    //用tuple装起来传入python
    args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyArray);
    //函数调用
    pRet = (PyArrayObject*)PyEval_CallObject(pFunc, args);
    emit sigColorMap();
    //qDebug()<<"(SocketServer::dataVisualization)emited signal!";
    return;
}

SocketServer::~SocketServer(){
    //释放内存
    // Py_CLEAR(pModule);
    // Py_CLEAR(pFunc);
    // Py_CLEAR(PyArray);
    // Py_CLEAR(args);
    // Py_CLEAR(pRet);
    // Py_Finalize();
    requestInterruption();
    quit();
    wait();
}
