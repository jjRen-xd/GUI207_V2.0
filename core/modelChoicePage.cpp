#include "modelChoicePage.h"
#include <QMessageBox>

using namespace std;

ModelChoicePage::ModelChoicePage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo):
    ui(main_ui),
    terminal(bash_terminal),
    modelInfo(globalModelInfo)
{
    // 模型类别选择框事件相应
    BtnGroup_typeChoice->addButton(ui->radioButton__TRA_DL__choice, 0);
    BtnGroup_typeChoice->addButton(ui->radioButton__FEA_RELE__choice, 1);
    BtnGroup_typeChoice->addButton(ui->radioButton__FEA_OPTI__choice, 2);
    BtnGroup_typeChoice->addButton(ui->radioButton__INCRE__choice, 3);
    connect(this->BtnGroup_typeChoice, &QButtonGroup::buttonClicked, this, &ModelChoicePage::changeType);

    // 确定
    connect(ui->pushButton_modelConfirm, &QPushButton::clicked, this, &ModelChoicePage::confirmModel);

    // 保存
    connect(ui->pushButton_saveModelAttri, &QPushButton::clicked, this, &ModelChoicePage::saveModelAttri);

    // 模型属性显示框
    attriLabelGroup["name"] = ui->lineEdit_modelChoice_name;
    attriLabelGroup["algorithm"] = ui->lineEdit_modelChoice_algorithm;
    attriLabelGroup["accuracy"] = ui->lineEdit_modelChoice_accuracy;
    attriLabelGroup["trainDataset"] = ui->lineEdit_modelChoice_trainDataset;
    attriLabelGroup["trainEpoch"] = ui->lineEdit_modelChoice_trainEpoch;
    attriLabelGroup["trainLR"] = ui->lineEdit_modelChoice_trainLR;
    attriLabelGroup["framework"] = ui->lineEdit_modelChoice_framework;
    attriLabelGroup["PATH"] = ui->lineEdit_modelChoice_PATH;
    attriLabelGroup["batch"] = ui->lineEdit_modelChoice_batch;
    attriLabelGroup["other"] = ui->lineEdit_modelChoice_other;

}

ModelChoicePage::~ModelChoicePage(){

}


void ModelChoicePage::changeType(){//选择模型类型
//    this->BtnGroup_typeChoice->checkedId()<<endl;
    // 获取选择的类型内容
    QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("__")[1];
    terminal->print("Selected Type: " + selectedType);

    // 更新下拉选择框
    vector<string> comboBoxContents = modelInfo->getNamesInType(
        selectedType.toStdString()
    );
    ui->comboBox_modelNameChoice->clear();
    for(auto &item: comboBoxContents){
        ui->comboBox_modelNameChoice->addItem(QString::fromStdString(item));
    }

}


void ModelChoicePage::confirmModel(bool notDialog = false){
    // 获取选择的类型内容
    QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("__")[1];
    modelInfo->selectedType = selectedType.toStdString(); // save type
    // 获取下拉框内容,即选择模型的名称
    QString selectedName = ui->comboBox_modelNameChoice->currentText();
    modelInfo->selectedName = selectedName.toStdString(); // save name
    terminal->print("Selected Type: " + selectedType + ", Selected Name: " + selectedName);

    if(!selectedType.isEmpty() && !selectedName.isEmpty()){
        // 更新属性显示标签
        updateAttriLabel();
        // 网络图像展示
        QString rootPath = QString::fromStdString(modelInfo->getAttri(modelInfo->selectedType,modelInfo->selectedName,"PATH"));
        QString imgPath = rootPath.split(".pt").first()+".png";
        terminal->print(imgPath);
        ui->label_modelImg->setPixmap(QPixmap(imgPath).scaled(QSize(400,400), Qt::KeepAspectRatio));

        if(!notDialog)
            QMessageBox::information(NULL, "模型切换提醒", "已成功切换模型为->"+selectedType+"->"+selectedName+"！");
    }
    else{
        if(!notDialog)
            QMessageBox::warning(NULL, "模型切换提醒", "模型切换失败，请指定模型");
    }
}


void ModelChoicePage::updateAttriLabel(){
    map<string,string> attriContents = modelInfo->getAllAttri(
        modelInfo->selectedType,
        modelInfo->selectedName
    );
    for(auto &currAttriWidget: this->attriLabelGroup){
        currAttriWidget.second->setText(QString::fromStdString(attriContents[currAttriWidget.first]));
    }
    ui->plainTextEdit_modelChoice_note->setPlainText(QString::fromStdString(attriContents["note"]));
}


void ModelChoicePage::saveModelAttri(){
    // 保存至内存
    string type = modelInfo->selectedType;
    string name = modelInfo->selectedName;
    if(!type.empty() && !name.empty()){
        string customAttriValue = "";
        // 对lineEdit组件
        for(auto &currAttriWidget: attriLabelGroup){
            customAttriValue = currAttriWidget.second->text().toStdString();
            if(customAttriValue.empty()){
                customAttriValue = "未定义";
            }
            this->modelInfo->modifyAttri(type, name, currAttriWidget.first, customAttriValue);
        }
        // 对plainTextEdit组件
        customAttriValue = ui->plainTextEdit_modelChoice_note->toPlainText().toStdString();
        if(customAttriValue.empty()){
            customAttriValue = "未定义";
        }
        this->modelInfo->modifyAttri(type, name, "note", customAttriValue);


        // 保存至.xml,并更新
        this->modelInfo->writeToXML(modelInfo->defaultXmlPath);
        this->confirmModel(true);

        // 提醒
        QMessageBox::information(NULL, "属性保存提醒", "模型属性修改已保存");
        terminal->print("模型："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改已保存");
    }
    else{
        QMessageBox::warning(NULL, "属性保存提醒", "属性保存失败，模型未指定！");
        terminal->print("模型："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改无效");
    }

}
