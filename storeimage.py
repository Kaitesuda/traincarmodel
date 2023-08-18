import base64
import pickle
import cv2
import requests  
import os

def imagetoVec(image):
    resized = cv2.resize(image,(128,128), cv2.INTER_AREA)
    v, buffer = cv2.imencode(".jpg", resized)
    image_str = base64.b64encode(buffer).decode('utf-8')
    imageDataString = "data:image/jpeg;base64," + image_str
    

    url = "http://127.0.0.1:8000/api/genhog"

    
    
    response = requests.get(url, json={"dataImage":imageDataString})
    
    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("JSON Decode Error:", e)
    else:
        print("API Request Error. Status Code:", response.status_code)
        return None
# img = cv2.imread("Cars Dataset/train/Audi/1.jpg")
# print(imagetoVec(img))
#dataPathImage="Cars Dataset/train"
dataPathImage="Cars Dataset/test"
X_list=[]
Y_list=[]

for sub in os.listdir(dataPathImage):
    sub_path = os.path.join(dataPathImage, sub)
    if os.path.isdir(sub_path):
        for fn in os.listdir(sub_path):   
            img_file_name = os.path.join(sub_path, fn)
            img = cv2.imread(img_file_name)
            X_list.append(img)
            Y_list.append(sub)
# for i in range(len(X_list)):
#     print(X_list[i]," ",Y_list[i])
HOGVectors=[]

for i in range(len(X_list)):
    try:
        res = imagetoVec(X_list[i])
        if res is not None and "HOG" in res:
            vec = res["HOG"]
            vec.append(Y_list[i])
            HOGVectors.append(vec)
        else:
            print("API response format error or missing 'HOG' key:", res)
    except Exception as e:
        print("Error processing image:", e)

#write_path_train="hogvectors_train.pkl"
write_path_test="hogvectors_test.pkl"

#pickle.dump(HOGVectors, open(write_path_train,"wb"))
pickle.dump(HOGVectors, open(write_path_test,"wb"))
print("done")