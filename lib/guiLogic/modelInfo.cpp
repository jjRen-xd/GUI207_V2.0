#include "modelInfo.h"

#include "./lib/guiLogic/tinyXml/tinyxml.h"

using namespace std;

ModelInfo::ModelInfo(string xmlPath):
    defaultXmlPath(xmlPath)
{
    loadFromXML(this->defaultXmlPath);
    this->selectedType = "";
    this->selectedName = "";

    // 中文名称对照表
    var2TypeName["TRA_DL"] = "深度学习模型";
    var2TypeName["FEA_RELE"] = "特征关联模型";
    var2TypeName["FEA_OPTI"] = "特征优化模型";
    var2TypeName["INCRE"] = "小样本增量学习模型";
    for(auto &item: var2TypeName){
        typeName2Var[item.second] = item.first;
    }
    // this->print();
}

ModelInfo::~ModelInfo(){
    this->clear();
}


vector<string> ModelInfo::getTypes(){
    vector<string> types;
    for(auto& it : infoMap) {
        types.push_back(it.first);
    }
    return types;
}


size_t ModelInfo::typeNum(){
    return infoMap.size();
}


vector<string> ModelInfo::getNamesInType(string type){
    vector<string> names;
    for(auto &item: this->infoMap[type]){
        names.push_back(item.first);
    }
    return names;
}


map<string,string> ModelInfo::getAllAttri(string Type, string Name){
    return infoMap[Type][Name];
}


string ModelInfo::getAttri(string type, string name, string attri){
    if (!checkMap(type,name,attri)) return "";
    return this->infoMap[type][name][attri];
}

void ModelInfo::modifyAttri(string Type, string Name, string Attri, string AttriValue){
        this->infoMap[Type][Name][Attri] = AttriValue;
}


void ModelInfo::print(){
    for(auto &datasetType: infoMap){
        cout<<"-->"<<datasetType.first<<endl;
        for(auto &datasetName: datasetType.second){
            cout<<"---->"<<datasetName.first<<endl;
            for(auto &datasetAttr: datasetName.second){
                cout<<"------>"<<datasetAttr.first<<"->"<<datasetAttr.second<<endl;
            }
        }
    }
}

void ModelInfo::clear(){
    infoMap.clear();
}


int ModelInfo::writeToXML(string xmlPath){
    TiXmlDocument *writeDoc = new TiXmlDocument; //xml文档指针
    TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");       //文档格式声明
    writeDoc->LinkEndChild(decl); //写入文档

    TiXmlElement *RootElement = new TiXmlElement("globalModelInfo");          //根元素
    RootElement->SetAttribute("datasetTypeNum", this->typeNum());  //属性
    writeDoc->LinkEndChild(RootElement);

    int typeID = 0;
    int nameID = 0;
    for(auto &datasetType: this->infoMap){  //n个父节点,即n个数据类型
        /* 对每个数据集类型建立节点 */
        typeID += 1;
        TiXmlElement *currTypeEle = new TiXmlElement(datasetType.first.c_str());
        currTypeEle->SetAttribute("typeID",typeID);         //设置节点属性
        RootElement->LinkEndChild(currTypeEle);             //父节点根节点

        //子元素
        for(auto &datasetName: datasetType.second){
            /* 对每个数据集建立节点 */
            nameID += 1;
            TiXmlElement *currNameEle = new TiXmlElement(datasetName.first.c_str());
            currTypeEle->LinkEndChild(currNameEle);
            currNameEle->SetAttribute("nameID",nameID);

            for(auto &datasetAttr: datasetName.second){
                /* 对每个属性建立节点 */
                TiXmlElement *currAttrEle = new TiXmlElement(datasetAttr.first.c_str());
                currNameEle->LinkEndChild(currAttrEle);

                TiXmlText *attrContent = new TiXmlText(datasetAttr.second.c_str());
                currAttrEle->LinkEndChild(attrContent);
            }
        }
    }

    writeDoc->SaveFile(xmlPath.c_str());
    delete writeDoc;

    return 1;
}

//解析xml文件
int ModelInfo::loadFromXML(string xmlPath){
    TiXmlDocument datasetInfoDoc(xmlPath.c_str());   //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the modelInfo file.Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        exit(1);
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info

    //遍历该结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){
        map<string, map<string,string>> datasetNameMap;

        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历子节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            map<string,string> datasetAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                datasetAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            datasetNameMap[currNameEle->Value()] = datasetAttrMap;
        }
        this->infoMap[currTypeEle->Value()] = datasetNameMap;
    }
    return 1;
}


int ModelInfo::addItemFromXML(string xmlPath){
    TiXmlDocument datasetInfoDoc(xmlPath.c_str());   //xml文档对象
    bool loadOk=datasetInfoDoc.LoadFile();                  //加载文档
    if(!loadOk){
        cout<<"Could not load the datasetInfo file.Error:"<<datasetInfoDoc.ErrorDesc()<<endl;
        return 0;
    }

    TiXmlElement *RootElement = datasetInfoDoc.RootElement();	//根元素, Info

    //遍历Type结点
    for(TiXmlElement *currTypeEle = RootElement->FirstChildElement(); currTypeEle != NULL; currTypeEle = currTypeEle->NextSiblingElement()){

        // 遍历节点属性
        TiXmlAttribute *pAttr=currTypeEle->FirstAttribute();
        while( NULL != pAttr){
            pAttr=pAttr->Next();
        }
        //遍历Name节点
        for(TiXmlElement *currNameEle=currTypeEle->FirstChildElement(); currNameEle != NULL; currNameEle=currNameEle->NextSiblingElement()){
            map<string,string> datasetAttrMap;
            // 遍历节点属性
            TiXmlAttribute *pAttr=currNameEle->FirstAttribute();
            while( NULL != pAttr){
                pAttr=pAttr->Next();
            }
            //遍历子子节点
            for(TiXmlElement *currAttrEle=currNameEle->FirstChildElement(); currAttrEle != NULL; currAttrEle=currAttrEle->NextSiblingElement()){
                datasetAttrMap[currAttrEle->Value()] = currAttrEle->FirstChild()->Value();
                // 遍历节点属性
                TiXmlAttribute *pAttr=currAttrEle->FirstAttribute();
                while( NULL != pAttr){
                    pAttr=pAttr->Next();
                }
            }
            this->infoMap[currTypeEle->Value()][currNameEle->Value()] = datasetAttrMap;
        }
    }
    return 1;
}


void ModelInfo::deleteItem(string type, string name){
    if(checkMap(type, name)){
        this->infoMap[type].erase(name);
    }
}

bool ModelInfo::checkMap(string type, string name, string attri){
    if(!this->infoMap.count(type)){
        return false;
    }
    else{
        if(name!="NULL" && !this->infoMap[type].count(name)){
            return false;
        }
        else{
            if(attri!="NULL" && !this->infoMap[type][name].count(attri)){
                return false;
            }
        }
    }
    return true;
}


//infoMap example
//map<QString, map<QString, map<QString,QString>>> infoMap = {
//    {"HRRP",{
//            {"HRRP_001",
//             {{"PATH","../../testPath"},{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"HRRP_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//    {"RCS",{
//            {"RCS_001",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"RCS_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//    {"Radia",{
//            {"Radia_001",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"Radia_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//    {"Image",{
//            {"Image_001",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            },
//            {"Image_002",
//             {{"claNum","5"},{"targetNumEachCla","1000"},{"pitchAngle","0"},{"azimuthAngle","[0, 90, 1]"},{"samplingNum","512"},{"incidentMode","水平入射"},{"freq","10GHz"},{"note","无"}}
//            }
//        },
//    },
//};
