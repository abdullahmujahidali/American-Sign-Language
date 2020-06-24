import cv2
import numpy as np
from tkinter import  *
from PIL import Image, ImageTk
root= Tk()
root.resizable(0,0)
root.geometry('200x150')     
root.title("American Sign Language")
photo = PhotoImage(file = "res.png")
root.iconphoto(False, photo)

def Run():
    # Loading  Yolo weights
    net = cv2.dnn.readNet("./Yolo/yolov3ASL.weights", "./Yolo/yolov3ASL.cfg")
    classes = list()
    f = open("./Yolo/yolov3ASL.names", "r")  # read .names file
    for line in f.readlines():  # loop each line in file
        classes.append(line.strip())
    layer_names = net.getLayerNames()  # get layes names
    layer = list()  # init layes list
    for i in net.getUnconnectedOutLayers():
        layer.append(layer_names[i[0] - 1])
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # provide video path in string or 0== internal webcam 2== external webcam
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")


    while(cap.isOpened()):
        # get video Frame by Frame
        ret, frame = cap.read()
        if ret == True:

            img = frame  # set img to frame.
            height, width, _ = img.shape
            blobObject = cv2.dnn.blobFromImage(
                img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blobObject)
            outputs = net.forward(layer)  # get next layer
            class_ids = list()
            confidences = list()
            boxes = list()
            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:  # tresh >0.5

                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # get points for rectangle
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            # number of dectected objects in frame
            dec = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in dec:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])  # get label
                    # each classs has specific color defined above
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                color, 2)  # create rectangle
                    cv2.putText(img, label, (10, 50), cv2.FONT_ITALIC, 2, color, 3)
            cv2.imshow("Image", img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


    cap.release()
    cv2.destroyAllWindows()

#class TestButtons:
#    def __init__(self,master):
#        frame= Frame(master)
#        frame.pack()
#        
#        self.printButton = Button(frame, text="Run Program", command=Run)
#        self.printButton.pack(side=LEFT)
#
#        self.quitButton = Button(frame, text="Quit", command=frame.quit)
#        self.quitButton.pack(side=LEFT)
#    def printMessage(self):
#        print("WOW")

whatever_you_do = " "
msg = Message(
    root, text = "Developed By\nAmna Ramzan.\nAbdullah Mujahid",
    fg="white",
    bg="black"
    )
msg.config(bg='Black', font=('Verdana', 18, 'bold'))
msg.pack()
b=Button(root,text="Click Here to Open Application !",command=Run)
b.pack()
quitButton = Button(root, text="Quit", command=root.quit)
quitButton.pack(side=LEFT)


root.mainloop()
