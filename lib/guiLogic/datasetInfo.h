#ifndef DATASETINFO_H
#define DATASETINFO_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

class DatasetInfo{
    public:
        DatasetInfo(std::string xmlPath);
        ~DatasetInfo();

        size_t typeNum();
        void print();
        void clear();

        std::vector<std::string> getTypes();                   // 获取所有的数据集类型
        std::vector<std::string> getNamesInType(std::string type); // 获取特定类型下的数据集名称
        std::string getAttri(std::string type, std::string name, std::string attri);
        std::map<std::string,std::string> getAllAttri(std::string Type, std::string Name);  // 获取指定是数据集的属性

        std::string defaultXmlPath;
        int writeToXML(std::string xmlPath);             // 将载入的数据集信息保存至.xml文件
        int loadFromXML(std::string xmlPath);            // 从.xml文件中读取所载入数据集的信息

        int addItemFromXML(std::string xmlPath);        // 从.xml文件中导入新数据集
        void deleteItem(std::string type, std::string name);

        std::string selectedType;
        std::string selectedName;
        std::vector<std::string> selectedClassNames;

        void modifyAttri(std::string Type, std::string Name, std::string Attri, std::string AttriValue);   //修改某一数据集的属性

        bool checkMap(std::string type, std::string name="NULL", std::string attri="NULL");

    private:

        // 所有数据集核心数据Map
        std::map<std::string, std::map<std::string, std::map<std::string,std::string>>> infoMap;
        // map<datasetType, map<datasetName, map<datasetAttri, attriValue>>>
};


#endif // DATASETINFO_H
