import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter #각 라벨마다 들어간 군집 개수 셀 때 이용
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa.display,librosa
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import time #코드 동작 소요시간 확인할 때

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

#check_perform_kmeans: cluster 수별 kmeans 성능 확인
def check_perform_kmeans(data):
    inertia = []
    K = range(1,16)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        inertia.append(kmeanModel.inertia_)
    #print (model.inertia_)

    # Plot the elbow
    plt.figure(figsize=(10, 7))
    plt.plot(K, inertia,'bx-')
    plt.xlabel('# of clusters')
    plt.ylabel('inertia')
    plt.show()

#calculate similarity
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def cluster_sim(labels,data,idx,num):
    sim_dic = {}

    for i in range(len(data.values)):
        seg1 = data.values[idx]  # 비교할 audio_slice
        seg2 = data.values[i]  # 같은 cluster에 있는 audio_slice 여러개

        seg1_2d = seg1.reshape(-1, 1)  # 차원 축소
        seg2_2d = seg2.reshape(-1, 1)

        sim = cos_sim(np.squeeze(seg1_2d), np.squeeze(seg2_2d))
        sim = sim * 100  # 퍼센트(%) 단위로 나타냄
        sim = round(sim, 2)  # 소수 둘째자리에서 반올림

        audio_slice_id = labels['fileName'].values[i]

        sim_dic[audio_slice_id] = sim

    final_dic = sorted(sim_dic.items(), reverse=True, key=lambda x: x[1])  # 내림차순 정렬

    return final_dic[1:num]

if __name__ == '__main__':
    # ""안에 파일 디렉토리 입력
    fileList = getFileList("C:/")
    data = getFeature("C:/", fileList)
    # Convert DataFrame to CSV
    data.to_csv("music_feature.csv", mode='w')

    # 여기서부터 유저가 선택한 노래 들어올 때 실행
    # Convert DataFrame to CSV and append single file
    new_data = getFeatureFile("C:/", "유저 파일이름")
    new_data.to_csv('music_feature.csv', mode='a', header=False)

    # Load music_feature.csv file
    feat_data = pd.read_csv("music_feature.csv", index_col=[0])

    # KMeans Clustering
    n_clusters = 10
    model = KMeans(n_clusters)
    model.fit(feat_data)
    data_labels = model.predict(feat_data)

    # cluster labeling 시각화
    fileList.append("유저 파일이름")
    print(len(data_labels), len(fileList))
    kmeans_labels = pd.DataFrame({"fileName": fileList, "Labels": data_labels})
    # cluster label 번호별로 필터링 하여 시각화
    filtering = (kmeans_labels['Labels'] == 1) #숫자 바꿔주기
    filter_kmeans_labels = kmeans_labels[filtering]

    # cluster label 번호별로 music feature csv파일 필터링
    filter_feat = feat_data.iloc[
        filter_kmeans_labels.index[filter_kmeans_labels['fileName'] == filter_kmeans_labels['fileName'].values[0:]]
    ]

    print(cluster_sim(filter_kmeans_labels,filter_feat,0, 6))