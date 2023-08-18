import cv2
import numpy as np
import os
import face_recognition

goryol="gorseller" #resimlerin bulunduğu klasör yolunu içerir.
resimler=[]
resimad=[]
liste=os.listdir(goryol) #bir dosya dızının  içeriğini döndürü
#print(liste)
for i in liste:
    res=cv2.imread(f'{goryol}/{i}')
    resimler.append(res)
    resimad.append(os.path.splitext(i)[0]) #os.path.splitext() fonksiyonu, bir dosya yolunun veya adının uzantısını ayırmak için kullanılır.
    """
    os.path.splitext() fonksiyonu bir dosya adının uzantısını ve dosya adını ayıran bir tuple (demet) döndürür.
     Bu tuple'ın ilk elemanı dosya adı (isim), ikinci elemanı ise dosya uzantısıdır.
    
    """
#print(resimler)
#print(resimad)
"""
gorseller" klasöründeki resimleri yükler, 
her resmin yüz kodlarını (kodlamalarını) bulur ve bu kodlamaları "bilinenencode" listesine ekler.
"""
def yuzkodbul(image):
    enodlist=[]
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        enodlist.append(encode)
    return enodlist
bilinenencode=yuzkodbul(resimler)
#print(len(bilinenencode))

cam=cv2.VideoCapture(0)

while True:
    _,frame=cam.read()
    frameR=cv2.resize(frame,(0,0),None,0.25,0.25) #boyutu küçültülür
    frameR=cv2.cvtColor(frameR,cv2.COLOR_BGR2RGB)
    faceloc=face_recognition.face_locations(frameR)
    #yüzlerin konumları ("faceloc") ve yüz kodlamaları ("encod") bulunur.
    encod=face_recognition.face_encodings(frameR,faceloc)

    for encodes,facelocs in zip(encod,faceloc):
        mathes=face_recognition.compare_faces(bilinenencode,encodes)
        dist=face_recognition.face_distance(bilinenencode,encodes)
        mindeger=np.argmin(dist)#dist dizisindeki en düşük değeri yani en küçük benzerlik uzaklığının dizindeki konumu bulunur.

        if mathes[mindeger]: # tespit edilen yüzün tanınan yüzlerle eşleşip eşleşmediğini kontrol eder.
            ad=resimad[mindeger].upper() #Eğer eşleşme varsa, ilgili yüzün adını ad adlı değişkene atar
            """
            resimad listesinden mindeger indeksine karşılık gelen yüz adını alır 
            ve büyük harfe çevirerek ad adlı değişkene atar
            """

            x,y,x1,y1=facelocs
            x,y,x1,y1= x*4,y*4,x1*4,y1*4
            cv2.rectangle(frame,(y1,x),(y,x1),(255,0,0),2)
            cv2.putText(frame,ad,(y1-16,x1+46),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
        else:
            cv2.putText(frame, "BULUNAMADI", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("webcam okuma",frame)
    if cv2.waitKey(1)==ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
"""
zip() fonksiyonu, verilen birden fazla liste veya diziyi paralel olarak birleştirip eşleştiren bir işlem yapar
np.argmin() fonksiyonu, verilen bir dizideki en küçük değerin indisini döndüren NumPy kütüphanesine ait bir fonksiyondur
"""