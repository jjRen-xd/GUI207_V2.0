#include "sensePage.h"
#include <QMessageBox>

#include <iostream>
#include <string>
#include <map>
#include <mat.h>

using namespace std;

SenseSetPage::SenseSetPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo)
{
    // 数据集类别选择框事件相应
    BtnGroup_typeChoice->addButton(ui->radioButton_HRRP_choice, 0);
    BtnGroup_typeChoice->addButton(ui->radioButton_RCS_choice, 1);
    BtnGroup_typeChoice->addButton(ui->radioButton_RADIO_choice, 2);
    BtnGroup_typeChoice->addButton(ui->radioButton_FEATURE_choice, 3);
    BtnGroup_typeChoice->addButton(ui->radioButton_IMAGE_choice, 4);
    connect(this->BtnGroup_typeChoice, &QButtonGroup::buttonClicked, this, &SenseSetPage::changeType);

    // 确定
    connect(ui->pushButton_datasetConfirm, &QPushButton::clicked, this, &SenseSetPage::confirmDataset);

    // 保存
    connect(ui->pushButton_saveDatasetAttri, &QPushButton::clicked, this, &SenseSetPage::saveDatasetAttri);

    // 下一批数据
    connect(ui->pushButton_nextSenseChart, &QPushButton::clicked, this, &SenseSetPage::nextBatchChart);

    // 数据集属性显示框
    this->attriLabelGroup["datasetName"] = ui->lineEdit_sense_datasetName;
    this->attriLabelGroup["claNum"] = ui->lineEdit_sense_claNum;
    this->attriLabelGroup["targetNumEachCla"] = ui->lineEdit_sense_targetNumEachCla;
    this->attriLabelGroup["pitchAngle"] = ui->lineEdit_sense_pitchAngle;
    this->attriLabelGroup["azimuthAngle"] = ui->lineEdit_sense_azimuthAngle;
    this->attriLabelGroup["samplingNum"] = ui->lineEdit_sense_samplingNum;
    this->attriLabelGroup["incidentMode"] = ui->lineEdit_sense_incidentMode;
    this->attriLabelGroup["freq"] = ui->lineEdit_sense_freq;
    this->attriLabelGroup["PATH"] = ui->lineEdit_sense_PATH;

    // 图片显示label成组
    imgGroup.push_back(ui->label_datasetClaImg1);
    imgGroup.push_back(ui->label_datasetClaImg2);
    imgGroup.push_back(ui->label_datasetClaImg3);
    imgGroup.push_back(ui->label_datasetClaImg4);
    imgGroup.push_back(ui->label_datasetClaImg5);
    imgGroup.push_back(ui->label_datasetClaImg6);

    imgInfoGroup.push_back(ui->label_datasetCla1);
    imgInfoGroup.push_back(ui->label_datasetCla2);
    imgInfoGroup.push_back(ui->label_datasetCla3);
    imgInfoGroup.push_back(ui->label_datasetCla4);
    imgInfoGroup.push_back(ui->label_datasetCla5);
    imgInfoGroup.push_back(ui->label_datasetCla6);

    // 显示图表成组
    chartGroup.push_back(ui->label_senseChart1);
    chartGroup.push_back(ui->label_senseChart2);
    chartGroup.push_back(ui->label_senseChart3);
    chartGroup.push_back(ui->label_senseChart4);
    chartGroup.push_back(ui->label_senseChart5);
    chartGroup.push_back(ui->label_senseChart6);

    chartInfoGroup.push_back(ui->label_senseChartInfo_1);
    chartInfoGroup.push_back(ui->label_senseChartInfo_2);
    chartInfoGroup.push_back(ui->label_senseChartInfo_3);
    chartInfoGroup.push_back(ui->label_senseChartInfo_4);
    chartInfoGroup.push_back(ui->label_senseChartInfo_5);
    chartInfoGroup.push_back(ui->label_senseChartInfo_6);


}

SenseSetPage::~SenseSetPage(){

}


void SenseSetPage::changeType(){
//    this->BtnGroup_typeChoice->checkedId()<<endl;
    // 获取选择的类型内容
    QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("_")[1];
    terminal->print("Selected Type: " + selectedType);

    // 更新下拉选择框
    vector<string> comboBoxContents = datasetInfo->getNamesInType(
        selectedType.toStdString()
    );
    ui->comboBox_datasetNameChoice->clear();
    for(auto &item: comboBoxContents){
        ui->comboBox_datasetNameChoice->addItem(QString::fromStdString(item));
    }

}


void SenseSetPage::confirmDataset(bool notDialog = false){
    // 获取选择的类型内容
    QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("_")[1];
    datasetInfo->selectedType = selectedType.toStdString(); // save type
    // 获取下拉框内容,即选择数据集的名称
    QString selectedName = ui->comboBox_datasetNameChoice->currentText();
    datasetInfo->selectedName = selectedName.toStdString(); // save name
    terminal->print("Selected Type: " + selectedType + ", Selected Name: " + selectedName);

    if(!selectedType.isEmpty() && !selectedName.isEmpty()){
        // 更新属性显示标签
        updateAttriLabel();

        // 绘制类别图
        for(int i = 0; i<6; i++){
            imgGroup[i]->clear();
            imgInfoGroup[i]->clear();
        }
        drawClassImage();

        ui->progressBar->setValue(40);

        // 绘制曲线
        for(int i=0;i<6;i++){
            if(!chartGroup[i]->layout()) delete chartGroup[i]->layout();
            chartInfoGroup[i]->clear();
            chartGroup[i]->clear();
        }
        nextBatchChart();

        // 绘制表格 TODO

        if(!notDialog)
            QMessageBox::information(NULL, "数据集切换提醒", "已成功切换数据集为->"+selectedType+"->"+selectedName+"！");
    }
    else{
        if(!notDialog)
            QMessageBox::warning(NULL, "数据集切换提醒", "数据集切换失败，请指定数据集");
    }
}


void SenseSetPage::updateAttriLabel(){
    map<string,string> attriContents = datasetInfo->getAllAttri(
        datasetInfo->selectedType,
        datasetInfo->selectedName
    );
    for(auto &currAttriWidget: this->attriLabelGroup){
        currAttriWidget.second->setText(QString::fromStdString(attriContents[currAttriWidget.first]));
    }
    ui->plainTextEdit_sense_note->setPlainText(QString::fromStdString(attriContents["note"]));
}


void SenseSetPage::drawClassImage(){
    string rootPath = datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
    // 寻找根目录下子文件夹的名称
    vector<string> subDirNames;
    dirTools->getDirs(subDirNames, rootPath);
    auto temp=std::find(subDirNames.begin(),subDirNames.end(),"model_saving");
    if(temp!=subDirNames.end()) subDirNames.erase(temp);
    datasetInfo->selectedClassNames = subDirNames; // 保存下，之后不用重复遍历

    for(int i = 0; i<subDirNames.size(); i++){
        imgInfoGroup[i]->setText(QString::fromStdString(subDirNames[i]));
        QString imgPath = QString::fromStdString(rootPath +"/"+ subDirNames[i] +".png");
        imgGroup[i]->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));
    }
}


void SenseSetPage::nextBatchChart(){
    string rootPath = datasetInfo->getAttri(datasetInfo->selectedType,datasetInfo->selectedName,"PATH");
    vector<string> subDirNames = datasetInfo->selectedClassNames;
    qDebug()<<"(SenseSetPage::nextBatchChart) subDirNames.size()="<<subDirNames.size();
    // 按类别显示
    for(int i=0; i<subDirNames.size(); i++){
        srand((unsigned)time(NULL));
        // 选取类别
        string choicedClass = subDirNames[i];
        string classPath = rootPath +"/"+ choicedClass;

        Chart *previewChart;

        vector<string> allMatFile;
        if(dirTools->getFiles(allMatFile, ".mat", classPath)){
            QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
            //下面这部分代码都是为了让randomIdx在合理的范围内（
            MATFile* pMatFile = NULL;
            mxArray* pMxArray = NULL;
            pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
            if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空！！！！！！";return;}
            std::string matVariable=allMatFile[0].substr(0,allMatFile[0].find_last_of('.')).c_str();//假设数据变量名同文件名的话

            QString chartTitle="Temporary Title";
            if(datasetInfo->selectedType=="HRRP") {chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";}
            else if (datasetInfo->selectedType=="RADIO") {chartTitle="RADIO Temporary Title";}
            else if (datasetInfo->selectedType=="RCS") {chartTitle="RCS Temporary Title";}
            pMxArray = matGetVariable(pMatFile,matVariable.c_str());
            if(!pMxArray){qDebug()<<"(ModelEvalPage::randSample)pMxArray变量没找到！！！！！！";return;}
            int N = mxGetN(pMxArray);  //N 列数
            int randomIdx = N-(rand())%N;

            //绘图
            previewChart = new Chart(ui->label_mE_chartGT,chartTitle,matFilePath);
            previewChart->drawImage(chartGroup[i],datasetInfo->selectedType,randomIdx);
            chartInfoGroup[i]->setText(QString::fromStdString(choicedClass+":Index")+QString::number(randomIdx));
        }


    }
}


void SenseSetPage::saveDatasetAttri(){
    // 保存至内存
    string type = datasetInfo->selectedType;
    string name = datasetInfo->selectedName;
    if(!type.empty() && !name.empty()){
        string customAttriValue = "";
        // 对lineEdit组件
        for(auto &currAttriWidget: attriLabelGroup){
            customAttriValue = currAttriWidget.second->text().toStdString();
            if(customAttriValue.empty()){
                customAttriValue = "未定义";
            }
            this->datasetInfo->modifyAttri(type, name, currAttriWidget.first, customAttriValue);
        }
        // 对plainTextEdit组件
        customAttriValue = ui->plainTextEdit_sense_note->toPlainText().toStdString();
        if(customAttriValue.empty()){
            customAttriValue = "未定义";
        }
        this->datasetInfo->modifyAttri(type, name, "note", customAttriValue);


        // 保存至.xml,并更新
        this->datasetInfo->writeToXML(datasetInfo->defaultXmlPath);
        this->confirmDataset(true);

        // 提醒
        QMessageBox::information(NULL, "属性保存提醒", "数据集属性修改已保存");
        terminal->print("数据集："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改已保存");
    }
    else{
        QMessageBox::warning(NULL, "属性保存提醒", "属性保存失败，数据集未指定！");
        terminal->print("数据集："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改无效");
    }

}
