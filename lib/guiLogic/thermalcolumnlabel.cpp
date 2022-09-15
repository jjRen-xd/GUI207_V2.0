#include "thermalcolumnlabel.h"

#include <QPainter>

ThermalColumnLabel::ThermalColumnLabel(QWidget *parent): QLabel(parent)
{

}
// 在控件发生重绘时触发的事件
void ThermalColumnLabel::paintEvent(QPaintEvent *)
{
    // 创建一个绘图对象，指定绘图设备为 QLabel
    QPainter painter(this);
    this->setBackgroundRole(QPalette::Base);
    this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    this->setScaledContents(true);
    this->setStyleSheet("border:2px solid red;");
    //线性渐变
    QLinearGradient linearGradient(QPointF(40, 190),QPointF(70, 190));
    //插入颜色
    linearGradient.setColorAt(0, Qt::yellow);
    linearGradient.setColorAt(0.5, Qt::red);
    linearGradient.setColorAt(1, Qt::green);
    // 绘制一个图像
    painter.setBrush(linearGradient);
    painter.drawRect(100, 100, 90, 40);
}
