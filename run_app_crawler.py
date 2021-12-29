import pyautogui as pg
import cv2
from pynput import mouse
import numpy as np
import os
import time
import threading as thread
import copy
# import easyocr
# import pandas as pd
import difflib
import json
import sys
import subprocess
import pytesseract
from scapy.all import *
from scapy.all import raw
import pyautogui as pg
import multiprocessing as mp
def http_header(packet):
    http_packet=str(packet)
    if http_packet.find('GET'):
        exit()

def GET_print(packet):

    packet_dict = {}
    for line in packet.show2(dump=True).split('\n'):
        if '###' in line:
            layer = line.strip('#[] ')
            packet_dict[layer] = {}
        elif '=' in line:
            key, val = line.split('=', 1)
            packet_dict[layer][key.strip()] = val.strip()
    # print(json.dumps(packet_dict))
    # print(hexdump(packet))

    # print(packet_dict)
    # print(pa)
    return 0
    # ret = "***************************************GET PACKET****************************************************\n"
    # ret += "\n".join(packet1.sprintf("{Raw:%Raw.load%}\n").split(r"\r\n"))
    # ret += "*****************************************************************************************************\n"
    # return ret

def worker():
    sniff(iface='Realtek PCIe GbE Family Controller #2',
          prn=http_header, filter="host legacy.hwahae.co.kr")



class scraping:
    def on_move(self, x, y):
        print('Position : x:%s, y:%s' % (x, y))

    def on_click(self, x, y, button, pressed):

        if (str(button) == "Button.right" and not pressed):
            if (self.productCnt < len(self.productPosition)):
                self.productPosition[self.productCnt] = (x, y)
                self.productCnt += 1
            elif (self.bboxRegion[0] is -1):
                self.bboxRegion[0] = x;
                self.bboxRegion[1] = y;
            elif (self.bboxRegion[2] is -1):
                self.bboxRegion[2] = x;
                self.bboxRegion[3] = y;
            elif (self.name is None):
                self.name = (x, y)
            elif (self.goback is None):
                self.goback = (x, y)


            print("-------가변적--------")
            print("제품의 이름의 위치들은",self.productPosition)
            print("-------고정적--------")
            print("제품 영역 imshow()", self.bboxRegion)
            print("제품명 위치",self.name)
            print("뒤로가기 버튼은",self.goback)
            print()
            if(self.goback is not None):
                self.saveJson()


    def on_scroll(self, x, y, dx, dy):
        # print('Scroll: (%s, %s) (%s, %s).' % (x, y, dx, dy))
        return False
    def setListener(self):
        with mouse.Listener(
                # on_move=self.on_move,
                on_click=self.on_click,
                on_scroll=self.on_scroll) as self.listener:
            self.listener.join()
    def saveJson(self):
        dic=dict()

        dic['productPosition'] = self.productPosition
        dic['bbox'] = self.bboxRegion
        dic['name'] = self.name
        dic['goback'] = self.goback

        with open("./tmp.json",'w') as w:
            json.dump(dic, w, indent='\t', ensure_ascii=False)
    def loadJson(self):
        with open('tmp.json', 'r') as f:
            dic = json.load(f)
            self.productPosition=dic['productPosition']
            self.bboxRegion=dic['bbox']
            self.name=dic['name']
            self.goback=dic['goback']
    def __removeText(self,img):
        from pytesseract import Output
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if (d['text'][i] != ""):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img[y:y + h, x:x + w] = 255
        return img
    def getLine(self,region,img):
        img=self.__removeText(copy.deepcopy(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img2 = copy.deepcopy(img).astype(np.uint8)
        edges = cv2.Canny(gray, 0, 10)
        # cv2.imshow("edge",edges)
        # cv2.waitKey(100000)
        x1, y1, x2, y2 = region
        minDot = x2 - x1 - 150

        h, w = img2.shape[:2]
        lines = cv2.HoughLines(edges, 1, np.pi / 180, minDot)
        y=[]
        if (lines is not None):
            for line in lines:  # 검출된 모든 선 순회
                r, theta = line[0]  # 거리와 각도
                tx, ty = np.cos(theta), np.sin(theta)  # x, y축에 대한 삼각비
                x0, y0 = tx * r, ty * r  # x, y 기준(절편) 좌표
                # 기준 좌표에 빨강색 점 그리기
                # cv2.circle(img2, (abs(x0), abs(y0)), 3, (0, 0, 255), -1)
                # 직선 방정식으로 그리기 위한 시작점, 끝점 계산
                x1, y1 = int(x0 + w * (-ty)), int(y0 + h * tx)
                x2, y2 = int(x0 - w * (-ty)), int(y0 - h * tx)
                # 선그리기
                cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
                y.append(y1)
            return (len(lines),img2,y)
        else:
            # 라인을 못찾은 경우
            return (None,img,y)

    def getPos(self):
        template=self.template
        # cv2.namedWindow("dddd", cv2.WINDOW_NORMAL)
        # cv2.imshow("dddd", template)
        # cv2.waitKey(0)
        pg.screenshot("temp2.png")
        src = cv2.imread("temp2.png", cv2.IMREAD_GRAYSCALE)

        result = cv2.matchTemplate(src, template, cv2.TM_SQDIFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        x, y = minLoc
        h, w = template.shape
        cx = int(w / 2 + x)
        cy = int(h / 2 + y)
        # cv2.namedWindow("dd",cv2.WINDOW_NORMAL)
        # cv2.imshow("dd",result)
        # cv2.waitKey(0)
        return (cx, cy)

    def showCurProduct(self,region):
        '''
        curProduct == -1
            초기 시작 단계
        curProduct == 0
            제품창이 아닌 다른 곳으로 이동한거
        :return:
        '''
        x1, y1, x2, y2 = region
        im1 = pg.screenshot("temp2.png", region=(x1, y1, x2 - x1, y2 - y1))
        self.curProduct = np.array(im1)
        x = x2 - x1 + 100
        x = x1 - x
        y = y1

        cv2.namedWindow("curProduct", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("curProduct", x, y)
        cv2.imshow("curProduct", self.curProduct)
        cv2.waitKey(1)

        pass

    def __init__(self, vP, iP,interval):
        self.interval=interval
        self.productPosition = [0 * 1 for i in range(vP)]
        self.bboxRegion = [-1 * 1 for i in range(4)]
        self.productCnt = 0
        self.name=None
        self.goback = None
        self.curProduct = -1;
        self.template=cv2.imread("templete.jpg",cv2.IMREAD_GRAYSCALE)


        print("start")

    def exceptionCheck(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
        sum=np.sum(img)
        rows,cols=img.shape
        return sum/(rows*cols)

    def inquireProductAfter5(self):
        self.showCurProduct(self.bboxRegion)
        check =self.exceptionCheck(self.curProduct)
        # print(line)

        x, y = self.productPosition[5]
        pg.moveTo(x, y)
        time.sleep(1)
        numberOfLine=3
        while (True and (check > 200)):
            self.showCurProduct(self.bboxRegion)
            # 예외체크
            self.exceptionCheck(self.curProduct)
            # 예외체크 끝
            line, img, pos = self.getLine(self.bboxRegion, self.curProduct)
            print(pos)
            cv2.imshow("curProduct", img)
            cv2.waitKey(1)
            # print("line", line)
            check2 = False
            while (True):
                if (line != None):
                    if (not (line >= numberOfLine)):
                        break
                self.showCurProduct(self.bboxRegion)
                check = self.exceptionCheck(self.curProduct)
                # print(check)
                if (check >= 200):
                    line, img, pos = self.getLine(self.bboxRegion, self.curProduct)
                    if (len(pos) >= numberOfLine):
                        minValue = min(pos)
                        maxValue = max(pos)
                        value = maxValue - minValue
                        rows, cols, _ = img.shape
                        if (value / rows < 0.8):
                            line = 0

                    cv2.imshow("curProduct", img)
                    cv2.waitKey(1)
                    x, y = self.productPosition[6]
                    pg.moveTo((x, y))
                    pg.dragTo(x, y - self.interval, 0.2, button='left')

                if (check < 200):
                    if (check2 is False):
                        check2 = True
                        start = time.time()
                    else:
                        elapsed = time.time() - start
                        if (elapsed > 4):
                            break
            check2 = False
            if (line is None):
                for i in range(2):
                    line, img, pos = self.getLine(self.bboxRegion, self.curProduct)
                    print(pos)
                    cv2.imshow("curProduct", img)
                    cv2.waitKey(1)
                    x, y = self.productPosition[6]
                    pg.moveTo((x, y))
                    pg.dragTo(x, y - self.interval, 0.2, button='left')
                    if (line is not None):
                        break

            while (True):
                if (line != None):
                    if (not (line < numberOfLine)):
                        break
                self.showCurProduct(self.bboxRegion)
                check = self.exceptionCheck(self.curProduct)
                # print(check)
                if (check >= 200):
                    line, img, pos = self.getLine(self.bboxRegion, self.curProduct)
                    print(pos)
                    cv2.imshow("curProduct", img)
                    cv2.waitKey(1)

                    x, y = self.productPosition[6]
                    pg.moveTo((x, y))
                    pg.dragTo(x, y - self.interval, 0.2, button='left')
                if (check < 200):
                    if (check2 is False):
                        check2 = True
                        start = time.time()
                    else:
                        elapsed = time.time() - start
                        if (elapsed > 4):
                            break

            if (check >= 200):
                #print("제품 선택")
                self.__start(self.productPosition[5])
                time.sleep(1)

        pass

    def __packetCheck(self, pos, name, click=1):
        print(name, "시작")
        check=True
        while (True):
            st = time.time()
            p = mp.Process(name=name, target=worker)
            p.start()
            if(check==True):
                check=False
                x, y = pos
                pg.click(x, y)
                pg.click(x, y)
                print("clicked")
            p.join(3)
            p.kill()
            elapsedTime = time.time() - st
            if (elapsedTime < 3):
                continue
            else:
                break
        print(name, "종료")
    def __endOfSales(self):
        '''
        아래 제품들이 단종된 경우 주석 해제하세요
        '''
        time.sleep(1)
        x,y=self.name
        pg.click(x,y)


    def __start(self, pos):
        self.cnt = 0
        # 제품 페이지,
        self.__packetCheck(pos=pos, name="SubProcess1")
        self.__packetCheck(pos=self.getPos(), name="SubProcess2")
        x, y = self.goback
        pg.click(x, y)

        print("뒤로가기1")

        time.sleep(1)
        x, y = self.goback
        pg.click(x, y)
        print("뒤로가기2")

        time.sleep(1)
    def __macroStart(self):

        # for i in range(6):
        #    self.__start(self.productPosition[i])
        self.inquireProductAfter5()
        pass
    def operation(self):
        self.setListener()
        self.__macroStart()
    def operation2(self):
        # self.setListener()
        self.loadJson()
        self.__macroStart()

if __name__ == "__main__":
    # sc = scraping(7, 9, 25)
    sc = scraping(7, 9, 24)
    sc.operation2()
    pass
