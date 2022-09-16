#include "realtimeinferencebuffer.h"
#include <QDebug>
RealTimeInferenceBuffer::RealTimeInferenceBuffer()
{

}
void RealTimeInferenceBuffer::put(std::vector<float> val)//生产物品
{

    std::unique_lock<std::mutex> lck(mtx);//unique_ptr

    while (!que.empty()){
        //que不为空，生产者应该通知消费者去消费，消费者消费完了，生产者再继续生产
        //生产者线程进入#1等待状态，并且#2把mtx互斥锁释放掉
        repo_not_full.wait(lck);//传入一个互斥锁，当前线程挂起，处于等待状态，并且释放当前锁 lck.lock()  lck.unlock
    }

    que.push(val);
    repo_not_empty.notify_all();

    qDebug()<<"(RealTimeInferenceBuffer::putputput) produce one";
}
std::vector<float> RealTimeInferenceBuffer::get()//消费物品
{
    //std::lock_guard<std::mutex> guard(mtx);//相当于scoped_ptr
    std::unique_lock<std::mutex> lck(mtx);//相当于unique_ptr 更安全
    while (que.empty()){
        //消费者线程发现que是空的，通知生产者线程先生产物品
        //#1 挂起，进入等待状态 #2 把互斥锁mutex释放
        repo_not_empty.wait(lck);
    }//如果其他线程执行notify了,当前线程就会从等待状态 =》到阻塞状态 =》但是要获取互斥锁才能继续向下执行
    std::vector<float> temp = que.front();
    que.pop();
    repo_not_full.notify_all();//通知其它线程我消费完了，赶紧生产吧
    qDebug()<<"(RealTimeInferenceBuffer::getgetget) consume one";
    return temp;
}
