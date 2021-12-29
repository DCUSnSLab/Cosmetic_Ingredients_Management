import json
import os
from tqdm import tqdm
def findStartWord(sentence):
    # 변환할 지점을 찾습니다.
    if (sentence.find('{') is not -1):
        return True
    else:
        return False


def getProductNames(txt):
    f = open("temp.json", 'w', encoding='UTF-8-sig')
    f.write(txt)
    f.close()
    with open('temp.json', 'r', encoding='UTF-8-sig') as f:
        sample = json.load(f)
    print(sample['brand'], sample['productName'])
    return (sample['brand'], sample['productName'])


def txt2Json(txt, pName):
    # print(txt)
    f = open("temp.json", 'w', encoding='UTF-8-sig')
    f.write(txt)
    f.close()
    newdict = dict(brand=pName[0], productName=pName[1])
    with open('temp.json', 'r', encoding='UTF-8-sig') as f:
        sample = json.load(f)
    # del sample['notificationText']
    newdict.update(sample)

    return newdict

def mkdir(path):
    path+="/jsonFiles/"
    if not os.path.exists(path):
        os.makedirs(path)

    # os.system("pause")
    return path

def removeProhibitionCharacters(fileName):
    lis=['\\','/',':','*','?','"','<','>','|']
    for i in lis:
        fileName=fileName.replace(i,"")
    return fileName


if __name__ == '__main__':
    products = "여성청결제"
    path='Z:/2021학년도/프로젝트/아토맘/'
    path+='데이터/헤어/'
    path+=products+'/'

    pList = []
    cnt = 0
    with open(products + '.txt', 'r', encoding='utf-8') as packetText:
        for i in packetText:
            if (findStartWord(i)):
                # data=getProductNames(i)
                pList.append(i)
            else:
                continue
    # print(len(pList))
    ingredients = pList[0::2]  # 짝수위치만
    products = pList[1::2]  # 홀수위치만
    length = len(products) + len(ingredients)
    result = dict()
    before = ""
    cnt = 0
    path=mkdir(path)
    for i in tqdm(range(len(products))):
    # for i in range(len(products)):
        cur = products[i]
        if cur != before:
            productData = getProductNames(products[i])
            dic = txt2Json(ingredients[i], productData)
            curProductName=productData[0]+'_'+productData[1]
            curProductName=curProductName.replace('\n',"")
            curProductName = curProductName.replace('\t', "")
            curProductName=removeProhibitionCharacters(curProductName)
            with open(path+curProductName+'.json', 'w', encoding='UTF-8-sig') as w:
               json.dump(dic, w, indent='\t', ensure_ascii=False)
            cnt += 1
        else:
            pass
            # print("중복 발생")
            # print(cur)
        before = cur

        result[i] = dic

    # print(result)
    print("총 제품 수 ", cnt)

