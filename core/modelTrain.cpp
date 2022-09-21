#include "modelTrain.h"

ModelTrain::ModelTrain(QTextBrowser* Widget, QLabel* trainImg, QLabel* valImg, QLabel* confusionMat, QProgressBar* trainProgressBar):
    OutShow(Widget),
    trainImg(trainImg),
    valImg(valImg),
    confusionMat(confusionMat),
    trainProgressBar(trainProgressBar),
    process_train(new QProcess(this)){
    QObject::connect(process_train, &QProcess::readyReadStandardOutput, this, &ModelTrain::readLogOutput);
    QObject::connect(process_train, &QProcess::readyReadStandardError, this, &ModelTrain::readLogError);
}

ModelTrain::~ModelTrain(){

}

void ModelTrain::readLogOutput(){
    /* 读取终端输出并显示 */
    QByteArray cmdOut = process_train->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        QStringList lines = logs.split("\n");
        int len=lines.length();
        for(int i=0;i<len;i++){
            QStringList Infos = lines[i].simplified().split(" ");
            if(lines[i].contains("Train Ending",Qt::CaseSensitive)){
                OutShow->append("===========================Train Ending===========================");
                showLog=false;
                showTrianResult();
            }
            else if(showLog){
                OutShow->append(lines[i]);
            }
        }
    }
    OutShow->update();
}

void ModelTrain::readLogError(){
    /* 读取终端Error并显示 */
    QByteArray cmdOut = process_train->readAllStandardError();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        QStringList lines = logs.split("\n");
        int len=lines.length();
        for(int i=0;i<len-1;i++){
            OutShow->append(lines[i]);
        }
    }
    OutShow->update();
    stopTrain();
}

void ModelTrain::startTrain(int modeltypeId,QString cmd){
  // TODO add code here
    modelTypeId = modeltypeId;
    if(process_train->state()==QProcess::Running){
        showLog=false;
        process_train->close();
        process_train->kill();
    }
    showLog=true;
    OutShow->setText("===========================Train Starting===========================");
    trainProgressBar->setMaximum(0);
    trainProgressBar->setValue(0);
    trainImg->clear();
    valImg->clear();
    confusionMat->clear();
    process_train->setProcessChannelMode(QProcess::MergedChannels);
    process_train->start("cmd.exe");
    process_train->write(cmd.toLocal8Bit() + '\n');
}

void ModelTrain::showTrianResult(){
    showLog=false;
    trainProgressBar->setMaximum(100);
    trainProgressBar->setValue(100);
    if(model_type==0){
        trainImg->setPixmap(QPixmap(saved_model_dir+"/training_accuracy.jpg"));
        valImg->setPixmap(QPixmap(saved_model_dir+"/verification_accuracy.jpg"));
        confusionMat->setPixmap(QPixmap(saved_model_dir+"/confusion_matrix.jpg"));
    }
    else if(model_type==1){
        trainImg->setPixmap(QPixmap(saved_model_dir+"/training_accuracy.jpg"));
        valImg->setPixmap(QPixmap(saved_model_dir+"/features_Accuracy.jpg"));
        confusionMat->setPixmap(QPixmap(saved_model_dir+"/confusion_matrix.jpg"));
    }
}

void ModelTrain::stopTrain(){
    showLog=false;
    trainProgressBar->setMaximum(100);
    trainProgressBar->setValue(0);
    OutShow->append("===========================Train Stoping===========================");
    if(process_train->state()==QProcess::Running){
        process_train->close();
        process_train->kill();
    }
}
