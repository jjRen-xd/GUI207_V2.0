#ifndef MODELCHOICEPAGE_H
#define MODELCHOICEPAGE_H

#include <map>
#include <vector>
#include <string>
#include <QObject>
#include <iostream>
#include <QButtonGroup>

#include "ui_MainWindow.h"
#include "./lib/guiLogic/bashTerminal.h"
#include "./lib/guiLogic/modelInfo.h"

class ModelChoicePage:public QObject{
    Q_OBJECT
public:
    ModelChoicePage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, ModelInfo *globalModelInfo);
    ~ModelChoicePage();

    QButtonGroup *BtnGroup_typeChoice = new QButtonGroup;
    std::map<std::string, QLineEdit*> attriLabelGroup;

public slots:
    void changeType();
    void confirmModel(bool notDialog);
    void saveModelAttri();
    void updateAttriLabel();

private:
    Ui_MainWindow *ui;
    BashTerminal *terminal;

    ModelInfo *modelInfo;
};

#endif // MODELCHOICEPAGE_H
