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

def getFileList(filePath):
    fileList = []
    for file in os.listdir(filePath):
        fileList.append(file)
    return fileList

# Extract Music Features: MFCC
# 처음에 training할 파일들의 feat 추출
def getFeatureAll(filePath, fileList):
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

def cluster_sim(data,idx,n):
    # idx:유사도 비교를 원하는 파일의 인덱스 번호,n:유사한 파일을 내림차순으로 상위 n개 보여줌
    sim_dic = {}

    for i in range(len(data.values)):
        seg1 = data.values[idx]  # 비교할 audio_slice
        seg2 = data.values[i]  # 같은 cluster에 있는 audio_slice 여러개

        seg1_2d = seg1.reshape(-1, 1)  # 차원 축소
        seg2_2d = seg2.reshape(-1, 1)

        sim = cos_sim(np.squeeze(seg1_2d), np.squeeze(seg2_2d))
        sim = sim * 100  # 퍼센트(%) 단위로 나타냄
        sim = round(sim, 2)  # 소수 둘째자리에서 반올림

        audio_slice_id = data.index[i]

        sim_dic[audio_slice_id] = sim

    final_dic = sorted(sim_dic.items(), reverse=True, key=lambda x: x[1])  # 내림차순 정렬

    return final_dic[1:n]

def similiarity(file):# 유사도 비교를 원하는 파일명(file)
    """
    # DB에 있는 모든 노래 feature구할 때 한번만 돌리기
    fileList = getFileList("C:/") #parameter:노래 디렉토리
    data = getFeatureAll("C:/", fileList)  #parameter:노래 디렉토리

    data.to_csv("music_feature.csv", mode='w') # Convert DataFrame to CSV
    """

    # 여기서부터 유저가 선택한 노래의 구간이 들어올 때 실행
    # Convert DataFrame to CSV and append single file
    new_data = getFeatureFile("C:/ffmpeg/bin/input/", file)
    new_data.to_csv('music_feature.csv', mode='a', header=False)

    # Load music_feature.csv file
    feat_data = pd.read_csv("music_feature.csv", index_col=[0])

    # KMeans Clustering(n_clusters가 군집 개수)
    n_clusters = 10
    model = KMeans(n_clusters)
    model.fit(feat_data)
    labels_data = model.predict(feat_data)
    #결과 값: [0,1,1,2,5,...]와 같은 형태로 나옴

    cluster_idx = [] #같은 군집으로 모여있는 파일의 인덱스를 모아놓은 리스트
    for i in range(len(labels_data)):
        # 새로 들어온 파일의 군집 번호와 일치할 때 cluster_idx로 append
        if (labels_data[i] == labels_data[len(feat_data) - 1]):
            cluster_idx.append(i)

    # 같은 군집으로 모여있는 파일의 feature만 필터링
    filter_feat_data = feat_data.iloc[cluster_idx]

    for i in range(len(filter_feat_data)):
        # 유사도 비교를 원하는 파일의 인덱스 번호 찾기(cluster_sim 함수 파라미터에 넣기 위해)
        if (filter_feat_data.index[i] == file):
            idx = i

    # 유사도 비교 함수 실행, 결과는 딕셔너리 타입
    # idx:유사도 비교를 원하는 파일의 인덱스 번호,상위 6개의 유사한 노래 보여줌
    print(cluster_sim(filter_feat_data, idx, 6))
    #결과 예시: [('4-TbQnONe_w_16.wav', 95.48), ('4-TbQnONe_w_19.wav', 95.42), ('4-TbQnONe_w_17.wav', 95.27),...]

if __name__ == '__main__':
    # 따옴표 안에 원하는 노래 파일명 삽입
    similiarity("Jason Derulo - Kiss The Sky(0).wav")
