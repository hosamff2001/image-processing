from tkinter import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import histogram as h
import histogram_equlization as e
import math
from skimage.util import random_noise
######################functions###################################
def order():
    ####################################Get the image and the factor############################################################
    old_image=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    fact=int(input("enter the factor : "))            
    ###########################################Set the attributes##############################################################################
    r,c,ch=old_image.shape
    new_row=r*fact
    new_column=c*fact
    new_image = np.zeros((new_row,new_column,ch))
    min=0
    max=0
    #############################################Put The original values ​​from the old image to the new image###########################################################################################
    r=range(0,r)
    c=range(0,c)
    ch=range(0,ch)
    for chh in ch:
        for row in r:
                for column in c:
                    new_image[fact*row,column*fact,chh]=old_image[row,column,chh]
    ####################################################fill the rows#############################################################################################
    for chh in ch:
        for row in r:
                for column in c:
                    i=1
                    if fact*(column+1)<new_column :
                        if(new_image[fact*row,fact*column,chh] <= new_image[fact*row,fact*(column+1),chh]):


                            min=new_image[fact*row,fact*column,chh]
                            max=new_image[fact*row,fact*(column+1),chh]

                                                                                                                #that for if the value in right is bigger than the left value
                            
                            spaces_array=range(fact*column+1 ,fact*(column+1))
                            for index in spaces_array:
                                new_image[fact*row,index,chh]=round(((max - min)/fact)*i + min)
                                i+=1
                        elif (new_image[fact*row,fact*column,chh] > new_image[fact*row,fact*(column+1),chh])  :

                            max=new_image[fact*row,fact*column,chh]
                            min=new_image[fact*row,fact*(column+1),chh]
                                                                                                                #that for if the value in left is bigger than the right value

                            spaces_array=range(fact*column+1 ,fact*(column+1))
                            for index in spaces_array:
                                new_image[fact*row,index,chh]=round(((max - min)/fact)*i + min)
                        
                                i+=1
                    
                    else :

                        new_image[fact*row,fact*column+1:new_column-1,chh]=new_image[fact*row,fact*column,chh]            #that for tha last value in row
                        break
  ####################################################fill the column#############################################################################################
    c=range(0,new_column)      
    for chh in ch:
        for row in r:                
                for column in c:
                    i=1
                    if  fact*(row+1)<new_row   :
                            if(new_image[fact*row,column,chh] <= new_image[fact*(row+1),column,chh] ):

                                
                                min=new_image[fact*row,column,chh]
                                max=new_image[fact*(row+1),column,chh]
                                            
                                                                                                        #that for if the value in right is bigger than the left value

                                spaces_array=range(fact*row+1 ,fact*(row+1))  
                                
                                for index in spaces_array:
                                    
                                    new_image[index,column,chh]=np.round(((max - min)/fact)*i + min)
                                    i+=1
                            elif new_image[fact*row,column,chh] > new_image[fact*(row+1),column,chh] :
                                
                                max=new_image[fact*row,column,chh]
                                min=new_image[fact*(row+1),column,chh]

                                                                                                                #that for if the value in left is bigger than the right value


                                spaces_array=range(fact*row+1 ,fact*(row+1))

                                for index in spaces_array:
                                    new_image[index,column,chh]=np.round(((max - min)/fact)*i + min)
                                    i+=1
                                
                    else :
                        new_image[fact*row+1:new_row-1,column,chh] = new_image[fact*row,column,chh]        #that for tha last value in column
    #################################################Show the old and new image##################################################################
    new_image=np.uint8(new_image)
    cv.imshow("Old Image",old_image)
    cv.imshow("New Image",new_image)
    cv.waitKey(0)

def watermark():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    logo=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgrams.png')
    r,c,ch=old.shape

    #################create the new image ##########################################
    new=np.zeros_like(old)
    

    #######################loop in old image and make the operations#########################33
    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                pixel=old[i,j,chh]  
                mask_pixel=pixel & 240  ##for make the low sintfic zeros by and with 240 =>1111 0000
                logo_pixel=logo[i,j,chh]
                logo_shift=logo_pixel>>4  ##for make the high sintfic zeros by shift 4 times right
                new_pexil=mask_pixel | logo_shift  ##or the result of previse operations
                new[i,j,chh]=new_pexil


    ###################show################################
    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)

def Guassian_function(image,sigma):
    # the smallest value = 0.5 any value less than 0.5 represent pixel opreation 1x1
    N = round(3.7*sigma-0.5)
    mask_size = 2*N+1

    image = cv.copyMakeBorder(image, N, N, N, N, cv.BORDER_REFLECT)
    t = round(mask_size/2)
    x= range(-t, t)
    filter = np.zeros([mask_size, mask_size], dtype=float)
    coef=(1/(2*np.pi*(sigma**2)))
    for i in range(mask_size):
        for j in range(mask_size):
            power=-((x[i]**2)+(x[j]**2))/(2*(sigma**2))
            temp=float(coef*np.exp(power))
            filter[i,j]=temp
    return filter,N

def normlization(old):
    r,c,ch=old.shape
    New_im = np.zeros((r,c,ch))
    max=0
    min=0
    
    for chh in range(0,ch):
        max=0
        min=old[0,0,chh]
        for row in range(0,r):
            for column in range(0,c):   
                if max<old[row,column,chh]:
                    max=old[row,column,chh]
                if min>old[row,column,chh]:
                    min=old[row,column,chh] 
        new_pexil=0
        for row in range(0,r) :
            for column in range(0,c): 
                new_pexil=((old[row,column,chh]-min)/(max-min))*255
                if new_pexil>255:
                    New_im[row,column,chh]=255
                elif new_pexil<0:
                    New_im[row,column,chh]=0
                else:
                    New_im[row,column,chh]=round(new_pexil)
                           
 
    New_im=np.uint8(New_im)
    return New_im

def Convert_to_Gray():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
  ###################take element############################
    r,c,ch=old.shape
    new = np.zeros((r,c,1), np.uint8)
 #####################insert values#######################
    new[:,:,0]=old[:,:,0]
 #################show##############################
    cv.imshow("old image",old)
    cv.imshow('new image',new)
    cv.waitKey(0)

def Drawing_the_histogram():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    ###########################set attrbute#########################################
    r,c,ch=old.shape
    colors_RED=np.zeros(256)
    colors_Green=np.zeros(256)
    colors_Blue=np.zeros(256)
    colors_range=range(0,256)
    fig = plt.figure()
    #################################put the numbers of pixels in ther postion in array######################### 
    for channal in range(0,ch):
        for row in range(0,r):
            for column in range(0,c):
                if channal ==0:   
                    colors_RED[old[row,column,channal]]+=1
                if channal==1:    
                    colors_Green[old[row,column,channal]]+=1
                if channal==2:
                        colors_Blue[old[row,column,channal]]+=1     
    ####################################show###############################################
        if channal  ==0:   
            plt1 = fig.add_subplot(221)
            plt1.plot(colors_range,colors_RED, color ='r')
        elif channal==1:    
            plt2 = fig.add_subplot(222)
            plt2.plot(colors_range,colors_Green, color ='g')
        elif channal==2:
            plt3 = fig.add_subplot(223)
            plt3.plot(colors_range,colors_Blue, color ='b')
    fig.subplots_adjust(hspace=.5,wspace=0.5)
    plt.show()

def Contrast():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    ####################take low and high contrast################################
    low=int(input("enter the low contrast :"))
    high=int(input("enter the high contrast :"))
    ###########################set attrbute#########################################
    r,c,ch=old.shape
    new = np.zeros((r,c,ch), np.uint8)
    r=range(0,r)
    c=range(0,c)
    ch=range(0,ch)
    #####################################search for the max and min ##########################################
    for chh in ch:
        min=255
        max=0
        for row in r:
                for column in c:
                    if old[row,column,chh] <min :
                        min=old[row,column,chh]
                    if old[row,column,chh] >max :
                        max=old[row,column,chh] 
    ############################################do the equation of contrast######################################
        for row in r:
                for column in c:
                     temp=(old[row,column,chh] - min)/(max - min) * (high - low) + low
                     if temp>255:
                         temp=255
                     if temp <0:
                         temp=0    
                     new[row,column,chh] = temp
    ###########################################show########################################################        
    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)

def Brightness():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    ####################take low and high contrast################################
    offset=int(input("enter the offset :"))
    ###########################set attrbute#########################################
    r,c,ch=old.shape
    new = np.zeros((r,c,ch), np.uint8)
    r=range(0,r)
    c=range(0,c)
    ch=range(0,ch)
    temp=0
    #################################add offset in the pixel##########################
    for chh in ch:
        for row in r:
                for column in c:
                    temp=old[row,column,chh]+offset
    ###########################cut off################################################################
                    if temp>255:
                        temp=255
                    new[row,column,chh]=temp   
    #################################show##########################################################        
    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)    

def power_low():
    ###########################start#######################
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    ###########################set attrbute#########################################
    gamma=float(input("enter the gamma :"))
    r,c,ch=old.shape
    new = np.zeros((r,c,ch))
    new_pexil=0
    #############################powe of gamma#############################################
    for chh in range(0,ch):
        for row in range(0,r):
            for column in range(0,c) :   
                new_pexil=255*(old[row,column,chh]/255)**gamma
                new[row,column,chh]=round(new_pexil)                          
    #####################call function contrast###################################   
    new=normlization(new)
    #################################show######################################################
    new=np.uint8(new)
    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)

def histogram_equalization():
    colors_rang=range(0,256)
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\2.jpeg')
    h.histogram(old)
    fig = plt.figure()    
    plt1 = fig.add_subplot(221)
    plt1.plot(colors_rang,h.color_Red, color ='r')
    plt2 = fig.add_subplot(222)
    plt2.plot(colors_rang,h.color_Green, color ='g')
    plt3 = fig.add_subplot(223)
    plt3.plot(colors_rang,h.color_Blue, color ='b')
    fig.subplots_adjust(hspace=.5,wspace=0.5)
    Sum_Red=np.zeros(256)
    Sum_Green=np.zeros(256)
    Sum_Blue=np.zeros(256)
    Sum_Red[0]=h.color_Red[0]
    Sum_Green[0]=h.color_Green[0]
    Sum_Blue[0]=h.color_Blue[0]
    for ch in range(0,3):
        for index in range(1,256):
            if ch==0:
                Sum_Red[index]=Sum_Red[index-1]+h.color_Red[index]
            elif ch==1:
                Sum_Green[index]=Sum_Green[index-1]+h.color_Green[index]
            elif ch==2:
                Sum_Blue[index]=Sum_Blue[index-1]+h.color_Blue[index]
    Sum_Red_max=Sum_Red.max()
    Sum_Green_max=Sum_Green.max()
    Sum_Blue_max=Sum_Blue.max()
    for ch in range(0,3):
        for index in range(0,256):
            if ch==0:
                Sum_Red[index]=round((Sum_Red[index]/Sum_Red_max)*255)
            elif ch==1:
                Sum_Green[index]=round((Sum_Green[index]/Sum_Green_max)*255)
            elif ch==2:
                Sum_Blue[index]=round((Sum_Blue[index]/Sum_Blue_max)*255)
    fig = plt.figure()
    plt1 = fig.add_subplot(221)
    plt1.plot(colors_rang,Sum_Red, color ='r')
    plt2 = fig.add_subplot(222)
    plt2.plot(colors_rang,Sum_Green, color ='g')
    plt3 = fig.add_subplot(223)
    plt3.plot(colors_rang,Sum_Blue, color ='b')
    fig.subplots_adjust(hspace=.5,wspace=0.5)
    plt.show()

def histogram_matching():
    matching_Red=np.zeros(256)
    matching_Green=np.zeros(256)
    matching_Blue=np.zeros(256)
    old1=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old1.shape
    old2=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgrams.png')
    e.histogram_equ(old1)
    Equ_1_Red=e.Sum_Red
    Equ_1_Green=e.Sum_Green
    Equ_1_Blue=e.Sum_Blue
    e.histogram_equ(old2)
    Equ_2_Red=e.Sum_Red
    Equ_2_Green=e.Sum_Green
    Equ_2_Blue=e.Sum_Blue
    for chh in range(0,3) :
        for index_match in range(0,256):
            if chh==0:
                min_diff=Equ_1_Red[0]
            elif chh==1:
                min_diff=Equ_1_Green[0]
            elif chh==2:
                min_diff=Equ_1_Blue[0]    
            i=0
            for index in range(0,256):
                if chh ==0 :
                    if abs(Equ_1_Red[index_match]- Equ_2_Red[index])==0 :
                        i=index
                        break
                    if abs(Equ_1_Red[index_match]- Equ_2_Red[index])< min_diff:
                        min_diff=  abs(Equ_1_Red[index_match]- Equ_2_Red[index])
                        i=index

                elif chh==1:
                    if abs(Equ_1_Green[index_match]- Equ_2_Green[index])==0 :
                        i=index
                        break
                    if abs(Equ_1_Green[index_match]- Equ_2_Green[index])< min_diff:
                        min_diff=  abs(Equ_1_Green[index_match]- Equ_2_Green[index])
                        i=index
                elif chh==2: 
                    if abs(Equ_1_Blue[index_match]- Equ_2_Blue[index])==0 :
                        i=index
                        break
                    if abs(Equ_1_Blue[index_match]- Equ_2_Blue[index])< min_diff:
                        min_diff=  abs(Equ_1_Blue[index_match]- Equ_2_Blue[index])
                        i=index 
            
            if chh==0 :                    
                matching_Red[index_match]=i 
            elif chh==1 :                    
                matching_Green[index_match]=i 
            elif chh==2 :                    
                matching_Blue[index_match]=i          

    for chh in range(0,ch):
        for row in range(0,r):
            for column in range(0,c) :
                if chh==0:
                    old1[row,column,chh]=matching_Red[old1[row,column,chh]]
                elif chh==1:
                    old1[row,column,chh]=matching_Green[old1[row,column,chh]]
                elif chh==2:
                    old1[row,column,chh]=matching_Blue[old1[row,column,chh]]    


    fig = plt.figure()

    colors_rang=range(0,256)
    plt1 = fig.add_subplot(221)
    plt1.plot(colors_rang,matching_Red, color ='r')

    plt2 = fig.add_subplot(222)
    plt2.plot(colors_rang,matching_Green, color ='g')

    plt3 = fig.add_subplot(223)
    plt3.plot(colors_rang,matching_Blue, color ='b')
    fig.subplots_adjust(hspace=.5,wspace=0.5)

    cv.imshow("old",old1)
    cv.waitKey(0)   
                    
    plt.show()

def add():  
    old1=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    old2=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgrams.png')

    ############################set attbuts#######################################
    r,c,ch=old1.shape
    new = np.zeros((r,c,ch))
    newpexil=0

    ###########################put addtion of two image#####################################
    for chh in range(0,ch):
        for row in range(0,r):
            for column in range(0,c) :   
                newpexil=old1[row,column,chh]+old2[row,column,chh]
                new[row,column,chh]=newpexil
                            
    #######################################call function########################################   
    new=normlization(new)
    new=np.uint8(new)
    cv.imshow('new',new)
    cv.waitKey(0)

def subtract():
    old1=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    old2=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgrams.png')

    ############################set attbuts#######################################
    r,c,ch=old1.shape
    new = np.zeros((r,c,ch))
    newpexil=0

    ###########################put addtion of two image#####################################
    for chh in range(0,ch):
        for row in range(0,r):
            for column in range(0,c) :   
                newpexil=old1[row,column,chh]-old2[row,column,chh]
                new[row,column,chh]=newpexil
                            
    #######################################call function########################################   
    new=normlization(new)
    new=np.uint8(new)
    cv.imshow('new',new)
    cv.waitKey(0)   

def negative():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape
    New_im = np.zeros((r,c,ch), np.uint8)
    r=range(0,r)
    c=range(0,c)
    ch=range(0,ch)
    temp=0
    for chh in ch:
        for row in r:
                for column in c:
                    temp=255-old[row,column,chh]  
                    if  temp<0:
                        temp=0
                    New_im[row,column,chh]=temp    
    cv.imshow("old",old)
    cv.imshow('new',New_im)
    cv.waitKey(0)    

def Quantization():
    
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    n=int(input("enter 1 to give Gray level or 2 to enter the number of bits per pixel (k) :"))

    if n==1 :
        Gray_level=int(input("enter the Gray level :"))
    elif n==2 :
        k=int(input("enter the number of bits per pixel :"))
        Gray_level=2**k
    else :
        print("enter wrong number!")
        exit()
    #################create the new image with orginal Dimensions##########################################
    r,c,ch=old.shape
    new = np.zeros((r,c,ch), np.uint8)

    ##################the old image given Border ###################################


    Gap=int(256/Gray_level)

    Colors= range(0,256,Gap)

    for channal in range(0,ch):
        for row in range(0,r):
            for column in range(0,c):
                temp=math.floor(old[row,column,channal]/Gap)
    
                new[row,column,channal] = Colors[temp]

    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)   

def Smoothing_with_Max_Filter():
    
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    N=int(input("enter the filter level :"))


    #################create the new image with orginal Dimensions##########################################
    r1,c1,ch1=old.shape
    new = np.zeros((r1,c1,ch1), np.uint8)

    ##################the old image given Border ###################################

    old=cv.copyMakeBorder(old, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    for chh in range(0,ch):
        for i in range(N,r-N):
            for j in range(N,c-N):
                temp=old[i-N:i+N+1,j-N:j+N+1,chh]
                value=round(np.max(temp))
                new[i-N,j-N,chh]=value

    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)    

def Smoothing_with_Mean_Filter():    
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    N=int(input("enter the filter level :"))


    #######################set attrbutes#####################################
    r1,c1,ch1=old.shape      
    new = np.zeros((r1,c1,ch1), np.uint8)     #################create the new image with orginal Dimensions



    ##################the old image given Border ###################################
    old=cv.copyMakeBorder(old, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape


    #######################loop of the old image ########################### 
    for chh in range(0,ch):
        for i in range(N,r-N):  
            for j in range(N,c-N):
                temp=old[i-N:i+N+1,j-N:j+N+1,chh]  
                value=round(np.sum(np.average(temp)))
                new[i-N,j-N,chh]=value


    #####################show##################3
    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)    

def Smoothing_with_Median_Filter():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    N=int(input("enter the filter level :"))


    #################create the new image with orginal Dimensions##########################################
    r1,c1,ch1=old.shape
    new = np.zeros((r1,c1,ch1), np.uint8)

    ##################the old image given Border ###################################

    old=cv.copyMakeBorder(old, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    for chh in range(0,ch):
        for i in range(N,r-N):
            for j in range(N,c-N):
                temp=old[i-N:i+N+1,j-N:j+N+1,chh]
                value=round(np.median(temp))
                new[i-N,j-N,chh]=value

    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)

def Smoothing_with_Min_Filter():
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    N=int(input("enter the filter level :"))


    #################create the new image with orginal Dimensions##########################################
    r1,c1,ch1=old.shape
    new = np.zeros((r1,c1,ch1), np.uint8)

    ##################the old image given Border ###################################

    old=cv.copyMakeBorder(old, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    for chh in range(0,ch):
        for i in range(N,r-N):
            for j in range(N,c-N):
                temp=old[i-N:i+N+1,j-N:j+N+1,chh]
                value=round(np.min(temp))
                new[i-N,j-N,chh]=value

    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)    

def Smoothing_with_weighted_Filter():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    sigma=float(input("enter the sigma :"))


    ##########################create new image with orginal dimensions###########
    r1, c1, ch1 = old.shape
    new = np.zeros((r1, c1, ch1), np.uint8)

    #######################create filter from sigma###################
    filter,N = Guassian_function(old,sigma)


    ##################the old image given Border ###################################
    old=cv.copyMakeBorder(old, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    #######################loop of the old image ###########################
    for chh in range(0, ch):
        for i in range(N, r - N):
            for j in range(N, c - N):
                temp = old[i - N:i + N + 1, j - N:j + N + 1, chh]
                result = np.multiply(temp, filter)
                value = round(np.sum(result))
                new[i - N, j - N, chh] = value


    ############################show##################################################            
    cv.imshow("old", old)
    cv.imshow('new', new)
    cv.waitKey(0)

def Edge_detection():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')



    ##########################create new image with orginal dimensions###########
    r1, c1, ch1 = old.shape
    new1 = np.zeros((r1, c1, 1), np.uint8)
    new2 = np.zeros((r1, c1, 1), np.uint8)
    new3 = np.zeros((r1, c1, 1), np.uint8)
    new4 = np.zeros((r1, c1, 1), np.uint8)
    #######################create filter from sigma###################
    filter1 =[[1,2,1],
            [0,0,0],
            [-1,-2,-1]]

    filter2 =[[1,0,-1],
            [2,0,-2],
            [1,0,-1]]

    filter3 =[[0,1,2],
            [-1,0,1],
            [-2,-1,0]]

    filter4 =[[2,1,0],
            [1,0,-1],
            [0,-1,-2]]


    ##################the old image given Border ###################################
    old=cv.copyMakeBorder(old, 1, 1, 1,1, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    #######################loop of the old image ###########################
    for chh in range(0, 1):
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                temp = old[i - 1:i + 1 + 1, j - 1:j + 1 + 1, chh]
                result1 = np.multiply(temp, filter1)
                value1 = round(np.sum(result1))
                new1[i - 1, j - 1, chh] = value1


                result2 = np.multiply(temp, filter2)
                value2 = round(np.sum(result2))
                new2[i - 1, j - 1, chh] = value2


                
                result3 = np.multiply(temp, filter3)
                value3 = round(np.sum(result3))
                new3[i - 1, j - 1, chh] = value3



                result4 = np.multiply(temp, filter4)
                value4 = round(np.sum(result4))
                new4[i - 1, j - 1, chh] = value4


    ####################################search for the max and min ##########################################
    for chh in range(0, 1):
        min1=255
        max1=0
        min2=255
        max2=0
        min3=255
        max3=0
        min4=255
        max4=0
        for row in range(0, r1):
                for column in range(0, c1):
                    if new1[row ,column ,chh] <min1 :
                        min1=new1[row ,column ,chh]
                    if new1[row,column,chh] >max1 :
                        max1=new1[row,column,chh]



                    if new2[row ,column ,chh] <min2 :
                        min2=new2[row ,column ,chh]
                    if new2[row,column,chh] >max2 :
                        max2=new2[row,column,chh]



                    if new3[row ,column ,chh] <min3 :
                        min3=new3[row ,column ,chh]
                    if new3[row,column,chh] >max3 :
                        max3=new3[row,column,chh] 



                    if new4[row ,column ,chh] <min4 :
                        min4=new4[row ,column ,chh]
                    if new4[row,column,chh] >max4 :
                        max4=new4[row,column,chh] 
    ############################################do the equation of contrast######################################
        for row in range(0, r1):
                for column in range(0, c1):
                    new1[row,column,chh] = (new1[row ,column  ,chh] - min1)/(max1 - min1) * (255 - 0) + 0


                    new2[row,column,chh] = (new2[row ,column  ,chh] - min2)/(max2 - min2) * (255 - 0) + 0


                    new3[row,column,chh] = (new3[row ,column  ,chh] - min3)/(max3 - min3) * (255 - 0) + 0


                    new4[row,column,chh] = (new4[row ,column  ,chh] - min4)/(max4 - min4) * (255 - 0) + 0

    ############################show##################################################            
    cv.imshow("old", old)
    cv.imshow('new1', new1)
    cv.imshow('new2', new2)
    cv.imshow('new3', new3)
    cv.imshow('new4', new4)


    cv.waitKey(0)

def Sharpening():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Desktop\images.jpeg')



    ##########################create new image with orginal dimensions###########
    r1, c1, ch1 = old.shape
    new1 = np.zeros((r1, c1, ch1), np.uint8)
    new2 = np.zeros((r1, c1, ch1), np.uint8)
    new3 = np.zeros((r1, c1, ch1), np.uint8)
    new4 = np.zeros((r1, c1, ch1), np.uint8)
    #######################create filter from sigma###################
    filter1 =[[0,1,0],
            [0,1,0],
            [0,-1,0]]

    filter2 =[[0,0,0],
            [1,1,-1],
            [0,0,0]]

    filter3 =[[1,0,0],
            [0,1,0],
            [0,0,-1]]

    filter4 =[[0,0,1],
            [0,1,0],
            [-1,0,0]]


    ##################the old image given Border ###################################
    old=cv.copyMakeBorder(old, 1, 1, 1,1, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    #######################loop of the old image ###########################
    for chh in range(0, ch1):
        for i in range(1, r - 1):
            for j in range(1, c - 1):
                temp = old[i - 1:i + 1 + 1, j - 1:j + 1 + 1, chh]
                result1 = np.multiply(temp, filter1)
                value1 = round(np.sum(result1))
                if value1>255:
                    value1=255
                if value1<0:
                    value1=0    
                new1[i - 1, j - 1, chh] = value1


                result2 = np.multiply(temp, filter2)
                value2 = round(np.sum(result2))
                if value2>255:
                    value2=255
                if value2<0:
                    value2=0      
                new2[i - 1, j - 1, chh] = value2


                
                result3 = np.multiply(temp, filter3)
                value3 = round(np.sum(result3))
                if value3>255:
                    value3=255
                if value3<0:
                    value3=0      
                new3[i - 1, j - 1, chh] = value3



                result4 = np.multiply(temp, filter4)
                value4 = round(np.sum(result4))
                if value4>255:
                    value4=255
                if value4<0:
                    value4=0  
                new4[i - 1, j - 1, chh] = value4


    ############################show##################################################            
    cv.imshow("old", old)
    cv.imshow('new1', new1)
    cv.imshow('new2', new2)
    cv.imshow('new3', new3)
    cv.imshow('new4', new4)


    cv.waitKey(0)

def Unsharpe():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    sigma=float(input("enter the sigma :"))


    ##########################create new image with orginal dimensions###########
    r1, c1, ch1 = old.shape
    new = np.zeros((r1, c1, ch1))

    #######################create filter from sigma###################
    filter,N = Guassian_function(old,sigma)


    ##################the old image given Border ###################################
    old=cv.copyMakeBorder(old, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    #######################loop of the old image ###########################
    for chh in range(0, ch):
        for i in range(N, r - N):
            for j in range(N, c - N):
                temp = old[i - N:i + N + 1, j - N:j + N + 1, chh]
                result = np.multiply(temp, filter)
                value = round(np.sum(result))
                
                new[i - N, j - N, chh] = value


    ############################unsharp#############################
    for chh in range(0, ch):
        for i in range(N, r - N):  
            for j in range(N, c - N):  
                new[i - N, j - N, chh] = old [i,j,chh] -new[i - N, j - N, chh] 
                if new[i - N, j - N, chh]<0 :
                    new[i - N, j - N, chh]=0
                new[i - N, j - N, chh] = old [i,j,chh] +new[i - N, j - N, chh] 
                if new[i - N, j - N, chh]>255 :
                    new[i - N, j - N, chh]=255
                



    # ############################show##################################################      
    new=np.uint8(new)      
    cv.imshow("old", old)
    cv.imshow('new', new)
    cv.waitKey(0)

def low_pass_idel():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape   


    #####################Apply FFT and FFTshift######################################
    dft = np.fft.fft2(old, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)


    #####################input from user##################################
    radius = int(input("enter the radius :"))  


    ############################make mask##################################
    mask = np.zeros_like(old)


    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                distance=int(((((i-(r/2))**2)+((j-(c/2))**2)) **0.5))
                if distance >radius :
                    mask[i,j,chh]=0 
                else :
                    mask[i,j,chh]=255


    #####################apply mask#######################################
    dft_shift_masked = np.multiply(dft_shift,mask) / 255


    #########################Inverse the shifting to return to the original form##############################
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)


    ###########################show###################################
    cv.imshow("ORIGINAL", old)
    cv.imshow("MASK", mask)
    cv.imshow("FILTERED IMAGE", img_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()

def low_pass_butterworth():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape   


    #####################Apply FFT and FFTshift######################################
    dft = np.fft.fft2(old, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)


    #####################input from user##################################
    radius = int(input("enter the radius :"))  
    n= int(input("enter the n :")) 


    ############################make mask##################################
    mask = np.zeros_like(old)
    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                distance=int(((((i-(r/2))**2)+((j-(c/2))**2)) **0.5))
                temp=(1/(1+((distance/radius))**(2*n)))*255
                mask[i,j,chh]=temp 


    #####################apply mask#######################################
    dft_shift_masked = np.multiply(dft_shift,mask) / 255


    #########################Inverse the shifting to return to the original form##############################
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)


    ######################################show############################################
    cv.imshow("ORIGINAL", old)
    cv.imshow("MASK", mask)
    cv.imshow("FILTERED IMAGE", img_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()

def low_pass_gaussian():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape   


    #####################Apply FFT and FFTshift######################################
    dft = np.fft.fft2(old, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)


    #####################input from user##################################
    radius = int(input("enter the radius :"))  


    ############################make mask##################################
    mask = np.zeros_like(old)

    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                distance=int(((((i-(r/2))**2)+((j-(c/2))**2)) **0.5))
                mask[i,j,chh]=int((math.exp(int(-pow(distance,2)/(2*pow(radius,2))))))
                mask[i,j,chh]=mask[i,j,chh]*255 


    #####################apply mask#######################################
    dft_shift_masked = np.multiply(dft_shift,mask) / 255


    #########################Inverse the shifting to return to the original form##############################
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)


    ###############################show#########################################
    cv.imshow("ORIGINAL", old)
    cv.imshow("MASK", mask)
    cv.imshow("FILTERED IMAGE", img_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()  

def high_pass_idel():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape   


    #####################Apply FFT and FFTshift######################################
    dft = np.fft.fft2(old, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)


    #####################input from user##################################
    radius = int(input("enter the radius :"))  


    ############################make mask##################################
    mask = np.zeros_like(old)


    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                distance=int(((((i-(r/2))**2)+((j-(c/2))**2)) **0.5))
                if distance >radius :
                    mask[i,j,chh]=255 
                else :
                    mask[i,j,chh]=0


    #####################apply mask#######################################
    dft_shift_masked = np.multiply(dft_shift,mask) / 255


    #########################Inverse the shifting to return to the original form##############################
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)


    ###########################show###################################
    cv.imshow("ORIGINAL", old)
    cv.imshow("MASK", mask)
    cv.imshow("FILTERED IMAGE", img_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()

def high_pass_butterworth():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape   


    #####################Apply FFT and FFTshift######################################
    dft = np.fft.fft2(old, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)


    #####################input from user##################################
    radius = int(input("enter the radius :"))  
    n= int(input("enter the n :")) 


    ############################make mask##################################
    mask = np.zeros_like(old)
    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                distance=int(((((i-(r/2))**2)+((j-(c/2))**2)) **0.5))
                temp=(1/(1+((distance/radius))**(2*n)))*255
                mask[i,j,chh]=255-temp 


    #####################apply mask#######################################
    dft_shift_masked = np.multiply(dft_shift,mask) / 255


    #########################Inverse the shifting to return to the original form##############################
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)


    ######################################show############################################
    cv.imshow("ORIGINAL", old)
    cv.imshow("MASK", mask)
    cv.imshow("FILTERED IMAGE", img_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()

def high_pass_gaussian ():
    old = cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    r,c,ch=old.shape   


    #####################Apply FFT and FFTshift######################################
    dft = np.fft.fft2(old, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)


    #####################input from user##################################
    radius = int(input("enter the radius :"))  


    ############################make mask##################################
    mask = np.zeros_like(old)

    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                distance=int(((((i-(r/2))**2)+((j-(c/2))**2)) **0.5))
                mask[i,j,chh]=int((math.exp(int(-pow(distance,2)/(2*pow(radius,2))))))
                mask[i,j,chh]=255-mask[i,j,chh]*255 


    #####################apply mask#######################################
    dft_shift_masked = np.multiply(dft_shift,mask) / 255


    #########################Inverse the shifting to return to the original form##############################
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
    img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)


    ###############################show#########################################
    cv.imshow("ORIGINAL", old)
    cv.imshow("MASK", mask)
    cv.imshow("FILTERED IMAGE", img_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()

def gaussian():
        
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\unnamed.png')
    x=int(input("enter 1 for Geometric mean or 2 for Mid-point :"))
    r1,c1,ch1=old.shape


    ######################adding noise#######################################
    old_with_nois=random_noise(old, mode='gaussian', seed=None,clip=True,)
    old_with_nois = np.array(255*old_with_nois, dtype = 'uint8')



    N=int(input("enter the filter level :"))


    #################create the new image with orginal Dimensions##########################################

    new = np.zeros((r1,c1,ch1), np.uint8)

    ##################the old image given Border ###################################

    old_with_nois=cv.copyMakeBorder(old_with_nois, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape

    ############################Geometric mean#########################
    if x== 1:
        for chh in range(0,ch):
            for i in range(N,r-N):
                for j in range(N,c-N):
                    temp=old_with_nois[i-N:i+N+1,j-N:j+N+1,chh]
                    sum=1
                    for i in range(2*N+1):
                        for j in range(2*N+1):
                            if temp[i,j]==0:
                                sum*=1
                            else :
                                sum*=temp[i,j]
                    root=np.power(2*N+1,2)   
                    value=np.power(sum,(1//root))
                    
                    new[i-N,j-N,chh]=int(value)

    ############################Mid-point#########################
    elif x==2: 
        for chh in range(0,ch):
            for i in range(N,r-N):
                for j in range(N,c-N):
                    temp=old_with_nois[i-N:i+N+1,j-N:j+N+1,chh]
                    new[i-N,j-N,chh]=((np.max(temp)+np.min(temp))//2)
            
    cv.imshow("old",old_with_nois)
    cv.imshow('new',new)
    cv.waitKey(0)    

def salt_and_pepper():
    
    #####################read image##########################################
    old=cv.imread(r'C:\Users\hosam\OneDrive\Pictures\Saved Pictures\instgram.png')
    N=int(input("enter the filter level :"))
    r1,c1,ch1=old.shape

    ########################adding noise####################################
    old_with_nois=random_noise(old, mode='s&p',amount=0.01)
    old_with_nois = np.array(255*old_with_nois, dtype = 'uint8')



    #################create the new image with orginal Dimensions##########################################

    new = np.zeros((r1,c1,ch1), np.uint8)

    ##################the old image given Border ###################################

    old_with_nois=cv.copyMakeBorder(old_with_nois, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    for chh in range(0,ch):
        for i in range(N,r-N):
            for j in range(N,c-N):
                temp=old_with_nois[i-N:i+N+1,j-N:j+N+1,chh]
                new[i-N,j-N,chh]=np.median(temp)
                

    cv.imshow("old",old_with_nois)
    cv.imshow('new',new)
    cv.waitKey(0)    


###########################gui#################################


window = Tk()
window.geometry("1200x400")  ##column x row
window.config(bg='#D8CBC2')
window.title("Image Processing GUI   By:Hossam ali")


# entry= Entry(window, width= 40)
# entry.focus_set()
# entry.pack()
l1 = Label(window, text = "Direct Mapping & \nWatermark")
l2 = Label(window, text = "Pixel Operations")
l3 = Label(window, text = "Smoothing Filter")
l4 = Label(window, text = "Sharping & \nEdge detection")
l5 = Label(window, text = "Frequency Domain \nFilter")
l6 = Label(window, text = "Image Restoration")
B0=Button(text="watermark ")
B0.config(command=watermark)
B1=Button(text="1_order")
B1.config(command=order)
B2=Button(text="convert to gray")
B2.config(command=Convert_to_Gray)
B3=Button(text="histogram")
B3.config(command=Drawing_the_histogram)
B4=Button(text="contrast")
B4.config(command=Contrast)
B5=Button(text="brightness")
B5.config(command=Brightness)
B6=Button(text="power low")
B6.config(command=power_low)
B7=Button(text="histogram equalzation")
B7.config(command=histogram_equalization)
B8=Button(text="histogram mathing")
B8.config(command=histogram_matching)
B9=Button(text="add")
B9.config(command=add)
B10=Button(text="subtract")
B10.config(command=subtract)
B11=Button(text="Negatives")
B11.config(command=negative)
B12=Button(text="Quantization")
B12.config(command=Quantization)
B13=Button(text="Smoothing with Max Filter")
B13.config(command=Smoothing_with_Max_Filter)
B14=Button(text="Smoothing with Mean Filter")
B14.config(command=Smoothing_with_Mean_Filter)
B15=Button(text="Smoothing with Median Filter")
B15.config(command=Smoothing_with_Median_Filter)
B16=Button(text="Smoothing with Min Filter")
B16.config(command=Smoothing_with_Min_Filter)
B17=Button(text="Smoothing with weighted Filter")
B17.config(command=Smoothing_with_weighted_Filter)
B18=Button(text="Edge detection")
B18.config(command=Edge_detection)
B19=Button(text="Sharpening")
B19.config(command=Sharpening)
B20=Button(text="Unsharpe")
B20.config(command=Unsharpe)
B21=Button(text="low pass idel")
B21.config(command=low_pass_idel)
B22=Button(text="low pass butterworth")
B22.config(command=low_pass_butterworth)
B23=Button(text="low pass gaussian")
B23.config(command=low_pass_gaussian)
B24=Button(text="high pass idel")
B24.config(command=high_pass_idel)
B25=Button(text="high pass butterworth")
B25.config(command=high_pass_butterworth)
B26=Button(text="high pass gaussian")
B26.config(command=high_pass_gaussian)
B27=Button(text="gaussian")
B27.config(command=gaussian)
B28=Button(text="salt&pepper")
B28.config(command=salt_and_pepper)


l1.config(font=('Monospace',15),bg='#D8CBC2',)
l2.config(font=('Monospace',15),bg='#D8CBC2',)
l3.config(font=('Monospace',15),bg='#D8CBC2',)
l4.config(font=('Monospace',15),bg='#D8CBC2',) 
l5.config(font=('Monospace',15),bg='#D8CBC2',)
l6.config(font=('Monospace',15),bg='#D8CBC2',)

B0.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B1.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B2.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B3.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B4.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B5.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B6.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B7.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B8.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B9.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B10.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B11.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B12.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B13.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B14.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B15.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B16.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B17.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B18.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B19.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B20.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B21.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B22.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B23.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B24.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B25.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B26.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B27.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))
B28.config(bg='#ff6200',fg='#fffb1f',activebackground='#FF0000',activeforeground='#fffb1f',font=('Ink Free',13,'bold'))






l1.grid(row = 0, column = 0, sticky = W)
l2.grid(row = 0, column = 1, sticky = W)
l3.grid(row = 0, column = 2, sticky = W)
l4.grid(row = 0, column = 3, sticky = W)
l5.grid(row = 0, column = 4, sticky = W)
l6.grid(row = 0, column = 5, sticky = W)

B0.grid(row = 1, column = 0, sticky = W)
B1.grid(row = 2, column = 0, sticky = W)



B2.grid(row = 1, column = 1, sticky = W)
B3.grid(row = 2, column = 1, sticky = W)
B4.grid(row = 3, column = 1, sticky = W)
B5.grid(row = 4, column = 1, sticky = W)
B6.grid(row = 5, column = 1, sticky = W)
B7.grid(row = 6, column = 1, sticky = W)
B8.grid(row = 7, column = 1, sticky = W)
B9.grid(row = 8, column = 1, sticky = W)
B10.grid(row = 9, column = 1, sticky = W)




B12.grid(row = 1, column = 2, sticky = W)
B13.grid(row = 2, column = 2, sticky = W)
B14.grid(row = 3, column = 2, sticky = W)
B15.grid(row = 4, column = 2, sticky = W)
B16.grid(row = 5, column = 2, sticky = W)
B17.grid(row = 6, column = 2, sticky = W)



B18.grid(row = 1, column = 3, sticky = W)
B19.grid(row = 2, column = 3, sticky = W)
B20.grid(row = 3, column = 3, sticky = W)



B21.grid(row = 1, column = 4, sticky = W)
B22.grid(row = 2, column = 4, sticky = W)
B23.grid(row = 3, column = 4, sticky = W)
B24.grid(row = 4, column = 4, sticky = W)
B25.grid(row = 5, column = 4, sticky = W)
B26.grid(row = 6, column = 4, sticky = W)




B27.grid(row = 1, column = 5, sticky = W)
B28.grid(row = 2, column = 5, sticky = W)
# B0.pack()
# B1.pack()
# B2.pack()
# B3.pack()
# B4.pack()
# B5.pack()
# B6.pack()
# B7.pack()
# B8.pack()
# B9.pack()
# B10.pack()
# B11.pack()
# B12.pack()
# B13.pack()
# B14.pack()
# B15.pack()
# B16.pack()
# B17.pack()
# B18.pack()
# B19.pack()
# B20.pack()
# B21.pack()
# B22.pack()
# B23.pack()
# B24.pack()
# B25.pack()
# B26.pack()
# B27.pack()
# B28.pack()
window.mainloop()