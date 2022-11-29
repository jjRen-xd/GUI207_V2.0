from xml.dom.minidom import Document

# 从txt读取每一行的隐层名
txtPath = "./modelLayers_densenet121.txt"
with open(txtPath, 'r') as f:
    innerLayers = f.readlines()
doc = Document()
modelName1 = doc.createElement(txtPath.split("/")[-1].split('_')[-1].split(".")[0])
doc.appendChild(modelName1)
modelName2 = doc.createElement(txtPath.split("/")[-1].split('_')[-1].split(".")[0])
modelName1.appendChild(modelName2)

for layer in innerLayers:
    layer = layer.replace("/", "_")
    # layerName = doc.createElement(layer.strip())
    nodeList = layer.split("_")
    for i in range(len(nodeList)):
        modeName = nodeList[i].strip()
        if modeName.isdigit():
            modeName = "_" + modeName

        if i == 0:
            # 如果以modeName为名的节点已经存在，就不再创建，直接挂
            if len(modelName2.getElementsByTagName(modeName)) == 0:
                node1 = doc.createElement(modeName)
                modelName2.appendChild(node1)
            else:
                node1 = modelName2.getElementsByTagName(modeName)[0]
        elif i == 1:
            if len(node1.getElementsByTagName(modeName)) == 0:
                node2 = doc.createElement(modeName)
                node1.appendChild(node2)
            else:
                node2 = node1.getElementsByTagName(modeName)[0]
        elif i == 2:
            if len(node2.getElementsByTagName(modeName)) == 0:
                node3 = doc.createElement(modeName)
                node2.appendChild(node3)
            else:
                node3 = node2.getElementsByTagName(modeName)[0]
        elif i == 3:
            if len(node3.getElementsByTagName(modeName)) == 0:
                node4 = doc.createElement(modeName)
                node3.appendChild(node4)
            else:
                node4 = node3.getElementsByTagName(modeName)[0]

    # Text = doc.createTextNode("None")
    # eval("node"+str(len(nodeList))).appendChild(Text)
    # layerName.appendChild(Text)

filename = "densenet121_hrrp_128_struct.xml"
f = open(filename, "w")
doc.writexml(f, addindent='\t', newl='\n', encoding="utf-8", standalone="yes")
f.close()