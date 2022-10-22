#ifndef SEARCHFOLDER_H
#define SEARCHFOLDER_H
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "./lib/guiLogic/tinyXml/tinyxml.h"
#include <dirent.h>//for opendir&mkdir

class SearchFolder{
    public:
        SearchFolder(){};
        ~SearchFolder(){};

        // 获取指定目录下的文件或文件夹名称
        bool getFiles(std::vector<std::string> &files, std::string filesType, std::string folderPath);
        bool getAllFiles(std::vector<std::string> &files, std::string folderPath);
        bool getDirs(std::vector<std::string> &dirs, std::string folderPath);
        
            // 判断文件是否存在
        bool exist(const std::string& name);
    private:

};


#endif // SEARCHFOLDER_H
