#ifndef REALTIMEINFERENCEBUFFER_H
#define REALTIMEINFERENCEBUFFER_H
#include <mutex>//互斥锁的头文件
#include <condition_variable>//条件变量的头文件
#include <queue>
#include <iostream>



//生产者生产一个物品，通知消费者消费一个；消费完了，消费者再通知生产者继续生产物品
class RealTimeInferenceBuffer
{
public:
    RealTimeInferenceBuffer();
    std::mutex mtx;//定义互斥锁，做线程间的互斥操作
    std::condition_variable repo_not_full;//条件变量指示产品缓冲区不满
    std::condition_variable repo_not_empty;//条件变量指示产品缓冲区不为空，就是缓冲区有产品
    void put(std::vector<float> val);
    std::vector<float> get();

private:
    std::queue<std::vector<float>> que;
};

#endif // REALTIMEINFERENCEBUFFER_H
