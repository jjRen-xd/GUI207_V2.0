#include "modelTrainPage.h"
#include <QMessageBox>
#include <QFileDialog>

ModelTrainPage::ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo):
    ui(main_ui),terminal(bash_terminal),datasetInfo(globalDatasetInfo),modelInfo(globalModelInfo){

    processTrain = new ModelTrain(ui->textBrowser, ui->train_img, ui->val_img, ui->confusion_mat, ui->timeRestEdit, ui->trainProgressBar);


    connect(ui->datadirButton, &QPushButton::clicked, this, &ModelTrainPage::chooseDataDir);
    connect(ui->starttrianButton, &QPushButton::clicked, this, &ModelTrainPage::startTrain);
    connect(ui->stoptrainButton, &QPushButton::clicked, this, &ModelTrainPage::stopTrain);

}


void ModelTrainPage::chooseDataDir(){
    QString dataPath = QFileDialog::getExistingDirectory(NULL,"请选择待训练数据的根目录","./",QFileDialog::ShowDirsOnly);
    if(dataPath == ""){
        QMessageBox::warning(NULL,"提示","未选择有效数据集根目录!");
        ui->datadirEdit->setText("");
        return;
    }
    ui->datadirEdit->setText(dataPath);
}

void ModelTrainPage::startTrain(){
    int modelType=ui->modeltypeBox->currentIndex();
//    QString modelType=ui->modeltypeBox->currentText();
    QString bathSize=ui->batchsizeBox->currentText();
    QString maxEpoch=ui->maxepochBox->currentText();
    QString dataDir = ui->datadirEdit->toPlainText();
    if(dataDir==""){
        QMessageBox::warning(NULL, "配置出错", "请指定待训练数据的根目录!");
        return;
    }
   //TODO 此处判断模型类型和数据是否匹配

   //TODO 此处判断模型类型和数据是否匹配
    ui->tabWidget->removeTab(0);
    ui->tabWidget->removeTab(1);
    ui->tabWidget->removeTab(2);
    if(modelType==0){
        ui->tabWidget->addTab(ui->tab,"训练集准确率");
        ui->tabWidget->addTab(ui->tab_2,"验证集准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
    }
    else if(modelType==1){
        ui->tabWidget->addTab(ui->tab_2,"特征准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
    }

    processTrain->startTrain(modelType, dataDir, bathSize, maxEpoch);
}

void ModelTrainPage::stopTrain(){
    processTrain->stopTrain();
}
