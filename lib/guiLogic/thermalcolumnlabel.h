#ifndef THERMALCOLUMNLABEL_H
#define THERMALCOLUMNLABEL_H

#include <QLabel>

class ThermalColumnLabel: public QLabel
{
public:
    ThermalColumnLabel(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *); // 重写绘图事件
};

#endif // THERMALCOLUMNLABEL_H
