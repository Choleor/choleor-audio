import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa.display,librosa
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import time #코드 동작 소요시간(사용 안해도 됨)

def getFileList(filePath):
    fileList = []
    for file in os.listdir(filePath):
        if(audio_slice_id==file):
            fileList.append(file)
    return fileList

"""
소요 시간 확인
start = time.time() 
print("time :", time.time() - start)
"""

# Extract Music Features: MFCC
# 처음에 training할 파일들의 feat 추출
def getFeature(filePath, fileList):
    X = pd.DataFrame()
    columns = []
    for i in range(20):
        columns.append("mfcc" + str(i + 1))

    for file in fileList:
        music, fs = librosa.load(filePath + str(file))
        mfcc_music = librosa.feature.mfcc(music, sr=fs)

        pca = PCA(n_components=1, whiten=True)  # use pca
        X_pca = pca.fit_transform(mfcc_music)
        fin_pca = []

        for index in range(len(X_pca)):
            fin_pca.append(X_pca[index, 0])
        df_pca = pd.Series(fin_pca)

        X = pd.concat([X, df_pca], axis=1)

        data = X.T.copy()
        data.columns = columns

    data.index = [file for file in fileList]

    return data

# 파일 하나의 feat 추출(user's input audio)
def getFeatureFile(filePath, file):
    X = pd.DataFrame()
    columns = []
    for i in range(20):
        columns.append("mfcc" + str(i + 1))

    music, fs = librosa.load(filePath + file)
    mfcc_music = librosa.feature.mfcc(music, sr=fs)

    pca = PCA(n_components=1, whiten=True)  # use pca
    X_pca = pca.fit_transform(mfcc_music)
    fin_pca = []

    for index in range(len(X_pca)):
        fin_pca.append(X_pca[index, 0])
    df_pca = pd.Series(fin_pca)

    X = pd.concat([X, df_pca], axis=1)

    data = X.T.copy()
    data.columns = columns
    data.index = [file]

    return data

"""
메소드 사용 예시
fileList=getFileList(파일 디렉토리)
new_data=getFeature(파일 디렉토리,fileList)
"""
# Convert DataFrame to CSV

#mode=write
new_data.to_csv("music_input_feature_mfcc.csv", mode='w')

#append single file
new_data.to_csv('music_input_feature_mfcc.csv',index=False, mode='a',header=False)

# Convert csv to DataFrame
feat_data=pd.read_csv("music_input_feature_mfcc.csv")

#KMeans Clustering
def execute_kmeans(data):
    n_clusters=10
    model = KMeans(n_clusters)
    model.fit(data)

    data_labels=model.predict(data)
    return data_labels

def check_perform_kmeans(music_data):
    inertia = []
    K = range(1,16)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(music_data)
        kmeanModel.fit(music_data)
        inertia.append(kmeanModel.inertia_)
    #print (model.inertia_)

    # Plot the elbow
    plt.figure(figsize=(10, 7))
    plt.plot(K, inertia,'bx-')
    plt.xlabel('# of clusters')
    plt.ylabel('inertia')
    plt.show()

#테스트 코드
execute_kmeans(feat_data)
print(len(execute_kmeans(feat_data)),len(fileList))
kmeans_labels = pd.DataFrame( { "fileName": fileList,"Labels": execute_kmeans(feat_data) } )

#cluster filtering
filtering=(kmeans_labels['Labels']==1)
filter_kmeans_labels=kmeans_labels[filtering]

#mfcc filtering
filter_feat_data=feat_data.iloc[filter_kmeans_labels.index[filter_kmeans_labels['fileName']==filter_kmeans_labels['fileName'].values[0:]]]

#calculate similarity
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def cluster_sim(idx, num):
    seg_sim_dic = {}

    for i in range(len(filter_feat_data.values)):
        seg1 = filter_feat_data.values[idx]  # 비교할 audio_slice
        seg2 = filter_feat_data.values[i]  # 같은 cluster에 있는 audio_slice들

        seg1_2d = seg1.reshape(-1, 1)  # 차원 축소
        seg2_2d = seg2.reshape(-1, 1)

        sim = cos_sim(np.squeeze(seg1_2d), np.squeeze(seg2_2d))
        sim = sim * 100  # 퍼센트(%) 단위로 나타냄
        sim = round(sim, 2)  # 소수 둘째자리에서 반올림

        audio_slice_id = filter_kmeans_labels['fileName'].values[i]

        seg_sim_dic[audio_slice_id] = sim

    final_dic = sorted(seg_sim_dic.items(), reverse=True, key=lambda x: x[1])  # 내림차순 정렬

    return final_dic[1:num]

if __name__ == '__main__':
    cluster_sim(0, 6)