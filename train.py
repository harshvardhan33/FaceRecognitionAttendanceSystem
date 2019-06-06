# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2019

@author: Harsh Vardhan
"""

from tkinter import *
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from itertools import count, cycle




            
window = Tk()
window.title("Face_Recogniser")
window.geometry('1280x720')
window.configure(background='black')

class ImageLabel(Label):    
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []
        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 1
        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()
    def unload(self):
        self.config(image=None)
        self.frames = None
    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)






message = Label(window, text="Face Recognition Attendance System" ,bg="black" ,fg="white"  ,width=50  ,height=3,font=("comicsansms",12,"bold"),borderwidth=3) 
message.pack(side = "top", fill = "both", expand = "no")

bottom = Label(window, text="" ,bg="Black" ,fg="white"  ,width=50  ,height=3) 
bottom.pack(side = "bottom", fill = "both", expand = "no")
lft = Label(window, text="" ,bg="Black" ,fg="white"  ,width=10  ,height=40) 
lft.pack(side = "left", fill = "both", expand = "no")
right = Label(window, text="" ,bg="Black" ,fg="white"  ,width=10  ,height=40) 
right.pack(side = "right", fill = "both", expand = "no")


imgbk = ImageLabel(window)
imgbk.pack(side = "bottom", fill = "both", expand = "yes")
imgbk.load('endit.gif')

lbl = Label(window, text="Enter ID",width=15  ,height=2  ,fg="white"  ,bg="#006bb3" ,font=("comicsansms",12,"bold"),relief="sunken") 
lbl.place(x=400, y=200)

txt = Entry(window,width=20,bg="#006bb3" ,fg="white",font=("comicsansms",12,"bold"),relief="sunken")
txt.place(x=700, y=215)

lbl2 = Label(window, text="Enter Name",width=15  ,fg="white"  ,bg="#006bb3"    ,height=2 ,font=("comicsansms",12,"bold"),relief="sunken") 
lbl2.place(x=400, y=300)

txt2 = Entry(window,width=20  ,bg="#006bb3"  ,fg="white",font=("comicsansms",12,"bold") ,relief="sunken" )
txt2.place(x=700, y=315)

lbl3 = Label(window, text="Notification : ",width=15  ,fg="white"  ,bg="#006bb3"  ,height=2 ,font=("comicsansms",12,"bold"),relief="sunken") 
lbl3.place(x=400, y=400)

message = Label(window, text="" ,bg="#006bb3"  ,fg="white"  ,width=30  ,height=2, activebackground = "yellow" ,font=("comicsansms",12,"bold"),relief="sunken") 
message.place(x=700, y=400)

lbl3 = Label(window, text="Attendance : ",width=20  ,fg="white"  ,bg="#006bb3"  ,height=2 ,font=("comicsansms",12,"bold")) 
lbl3.place(x=400, y=650)


message2 = Label(window, text="" ,fg="white"   ,bg="#006bb3",activeforeground = "green",width=30  ,height=4  ,font=("comicsansms",12,"bold"),relief="sunken") 
message2.place(x=700, y=650)


def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)

  
clearButton = Button(window, text="Clear", command=clear  ,fg="White"  ,bg="#006bb3"  ,width=15  ,height=1 ,activebackground = "Red" ,font=("comicsansms",12,"bold"),relief="sunken")
clearButton.place(x=970, y=200)
clearButton2 = Button(window, text="Clear", command=clear2  ,fg="White"  ,bg="#006bb3"  ,width=15  ,height=1, activebackground = "Red" ,font=("comicsansms",12,"bold"),relief="sunken")
clearButton2.place(x=970, y=300)    
takeImg = Button(window, text="Take Images", command=TakeImages  ,fg="White"  ,bg="#006bb3"  ,width=15  ,height=2, activebackground = "Red" ,font=("comicsansms",12,"bold"),relief="sunken")
takeImg.place(x=200, y=500)
trainImg = Button(window, text="Train Images", command=TrainImages  ,fg="White"  ,bg="#006bb3"  ,width=15  ,height=2, activebackground = "Red" ,font=("comicsansms",12,"bold"),relief="sunken")
trainImg.place(x=500, y=500)
trackImg = Button(window, text="Track Images", command=TrackImages  ,fg="White"  ,bg="#006bb3"  ,width=15  ,height=2, activebackground = "Red" ,font=("comicsansms",12,"bold"),relief="sunken")
trackImg.place(x=800, y=500)
quitWindow = Button(window, text="Quit", command=window.destroy  ,fg="White"  ,bg="#006bb3"  ,width=15  ,height=2, activebackground = "Red" ,font=("comicsansms",12,"bold"),relief="sunken")
quitWindow.place(x=1100, y=500)
copyWrite = Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)


copyWrite.configure(state="disabled",fg="red"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)


window.mainloop()