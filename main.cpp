#include "./core/MainWindow.h"
#include "./lib/guiLogic/tools/guithreadrun.h"
#include <QApplication>

int main(int argc, char *argv[]){
    QApplication a(argc, argv);
    GuiThreadRun::inst();  		// 在Gui线程创建GuiThreadRun全局对象
    MainWindow w;
    w.show();
    return a.exec();
}
