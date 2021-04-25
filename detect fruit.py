
import cv2
import numpy as np
import sys
import serial

ser=serial.Serial('COM15',baudrate=9600)
cap = cv2.VideoCapture(0)
# kiem tra version opencv  : cv2.__version__

    
while(cap.isOpened()):
    
    # doc video
    ret, frame = cap.read()
    # loc nhieu
    frame = cv2.medianBlur(frame,1)
    # Resize kich thuoc moi truc 20%
    frame=cv2.resize(frame,(0,0), fx=1, fy=1.5)
    
    # Roi image chieu dai truoc, toi chieu ngang, (tu diem nay den dem kia)
    frame=frame[0:640,90:495]
    '''
    print(frame.shape)
    '''

    # chuyen sang anh xam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # de inrang theo mau hsv can chuyen tu bgr => hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # Nguong min va max de Inrang
    lower_range = np.array([10, 0, 0], dtype=np.uint8) 
    upper_range = np.array([89, 255, 255], dtype=np.uint8)

    # Inrang theo nguong tren , truyen vao hsv
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cv2.imshow("inrang",mask)

    # tinh kich thuoc cua trai bang cach dem diem mau trang
    '''
    wheit =cv2.countNonZero(mask)
    '''
    #print(wheit)
###################### so sanh mau#################
    # tao 2 nguong de so sanh
    
    '''
    lower = {'xanh':(33, 27, 74), 'vang':(25, 113, 156),'bihu':(0, 86, 97)} 
    upper = {'xanh':(96, 244, 244), 'vang':(33, 223, 255),'bihu':(30, 178, 244)}
   
    # chuan de so sanh
    colors = {'xanh':(65,135,159), 'vang':(29,168,205),'bihu':(15, 132, 170) }

    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
       # kernel = np.ones((9,9),np.uint8)
        mau = cv2.inRange(hsv, lower[key], upper[key])
        cv2.imshow("mau",mau)
       # mau = cv2.morphologyEx(mau, cv2.MORPH_OPEN, kernel)
       # mau = cv2.morphologyEx(mau, cv2.MORPH_CLOSE, kernel)

        '''
####################################################
    # giong nhu khu nhieu truyen vao anh xam
    mask = cv2.erode(mask, None, iterations=2)
    # Lam ro anh hon truyen vao anh xam
    mask = cv2.dilate(mask, None, iterations=2)
    
    # contour ra bien dang cua hinh tron
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(mask,50,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # contour theo canny
    '''
    canny_output = cv2.Canny(gray, 50, 50 * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    # xet them dieu kien dien tich   - neu dien tich lon hon bao nhiu do thi cho no se bo
    
    # Xac dinh toa do cua cac diem contour - khi nao contour nhieu thi cho chay trong for
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])

        # xac dinh tam cua trai chanh
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
        area=cv2.contourArea(contours[i])
        if area > 30000:
            # xac dieu kien khi tam no lon hon 640/2  va be hon 640/2+5 la trai chanh nam o giau 
            # thi se gui du lieu di ngoai vung nay thi khong gui du lieu
            if mc[i][1]>320 and mc[i][1]<330:
                # gui di dien tich, mau, toa do luc gui di 
                # ve tam cua trai chanh
                cv2.circle(frame, (int(mc[i][0]), int(mc[i][1])), 4, (0,0,255), -1)
                #- ve contour len frame
                cv2.drawContours(frame, contours, -1, (0,255,0), 3)

                boundaries = [
                ('bihu',[0, 86, 97], [25, 178, 244]),
                ('vang',[27, 113, 156], [33, 223, 255]),
                ('xanh',[33, 54, 74], [78, 253, 244]),
                ]
                for (key,lower, upper) in boundaries:
                    lower = np.array(lower, dtype = "uint8")
                    upper = np.array(upper, dtype = "uint8")
                    mask = cv2.inRange(hsv, lower, upper)
                    pixeltrang =cv2.countNonZero(mask)
                    if pixeltrang>30000:
                        print(area)
                        #print(mc[i][0])
                        #print(mc[i][1])
                        #cv2.imshow("mau",mask)
                        if area> 40000 and area<60000 :
                            if key=='xanh':
                                ser.write('1'.encode()) 
                                print('xanh nho')
                            elif key=='vang':
                                ser.write('3'.encode()) 
                                print('vang nho')
                            else:
                                ser.write('5'.encode()) 
                                print('bi hu')

                        else:
                            if key=='xanh':
                                ser.write('2'.encode()) 
                                print('xanh to')
                            elif key=='vang':
                                ser.write('4'.encode()) 
                                print('vang to')
                            else:
                                ser.write('5'.encode()) 
                                print('bi hu')

                                



    #ve ca tam 
    #drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
    #for i in range(len(contours)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        #cv2.drawContours(drawing, contours, i, color, 2)
        
    # truyen vao anh xam => tim ra hinh tron
    #tim hinh tron ve hinh tron
    '''
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=65,param2=30,minRadius=5,maxRadius=1000)   
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle tren anh frame
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
            # draw the center of the circle tren anh frame
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            print(i[0])
            print(i[1])
    '''
    # tim ra nguong mau hsv de inrang
    #blue = sys.argv[1]
    #green = sys.argv[2]r
    #red = sys.argv[3]  
    '''
    color = np.uint8([[[71, 234, 213]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    
    hue = hsv_color[0][0][0]
    
    print("Lower bound is :"),
    print("[" + str(hue-10) + ", 100, 100]\n")
    
    print("Upper bound is :"),
    print("[" + str(hue + 10) + ", 255, 255]")
    '''

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()