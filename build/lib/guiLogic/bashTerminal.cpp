#include "bashTerminal.h"
#include <ctime>
#include <chrono>

BashTerminal::BashTerminal(QLineEdit *inWidget, QTextEdit *outWidget):
    bashInEdit(inWidget),
    bashOutShow(outWidget),
    process_bash(new QProcess(this)){  

    process_bash->start(bashApi);
    inWidget->setPlaceholderText("Your command");
    bashOutShow->setLineWrapMode(QTextEdit::NoWrap);
    process_bash->setProcessChannelMode(QProcess::MergedChannels);
    QObject::connect(process_bash, &QProcess::readyReadStandardOutput, this, &BashTerminal::readBashOutput);
    QObject::connect(process_bash, &QProcess::readyReadStandardError, this, &BashTerminal::readBashError);

}


BashTerminal::~BashTerminal(){
    process_bash->terminate();
}


void BashTerminal::print(QString msg){
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    tm* tm_time = localtime(&nowTime);
    char timefn[128] = { 0 };
    strftime(timefn,128,"%a,%d %b %Y %X   ",tm_time);
    QString timef=timefn;

    bashOutShow->append(timef+msg);
    bashOutShow->update();
}


void BashTerminal::runCommand(QString cmd){
    /* 开放在终端运行命令接口 */
    bashOutShow->append("Shell:~$ "+cmd);
    process_bash->write(cmd.toLocal8Bit() + '\n');
}


void BashTerminal::commitBash(){
    /* 在GUI上手动输入向终端提交命令 */
    QString cmdIn = bashInEdit->text();
    bashOutShow->append("Shell:~$ "+cmdIn);
    process_bash->write(cmdIn.toLocal8Bit() + '\n');
    bashInEdit->clear();
}


void BashTerminal::cleanBash(){
    /* 清空并重启终端 */
    bashOutShow->clear();
    process_bash->close();
    process_bash->start("powershell");
}


void BashTerminal::readBashOutput(){
    /* 读取终端输出并显示 */
    QByteArray cmdOut = process_bash->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        bashOutShow->append(QString::fromLocal8Bit(cmdOut));
    }
    bashOutShow->update();
}


void BashTerminal::readBashError(){
    /* 读取终端Error并显示 */
    QByteArray cmdOut = process_bash->readAllStandardError();
    if(!cmdOut.isEmpty()){
        bashOutShow->append(QString::fromLocal8Bit(cmdOut));
    }
    bashOutShow->update();
}
