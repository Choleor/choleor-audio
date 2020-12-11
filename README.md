# choleor_audio
Audio Server for choleor service, Choreography Generator Service<br>
Our service makes choreography depending on similarity of other music.<br><br>

다음 내용은 audio 서버에서 수행하는 안무 후보 추천 로직 중 일부이다.

## Similarity of other music(비지도학습을 이용하여 유사한 음원끼리 묶기)
음악의 feature를 분석해서 feature를 숫자로 나타낸 데이터값을 기반으로 여러 음원 파일들을 학습하여 비교한다. Chromagram,MFCC feature를 위주로 하였으며 그 중 mfcc를 바탕으로 추출한 다음 normalize를 위해 dimension reduction을 수행한다. 방법은 PCA(주성분 분석)을 이용하였다.

구간 파일 하나마다 feature 분석후 pca로 차원을 줄이는 과정이다.

```
import librosa.display,librosa
from sklearn.decomposition import PCA
import pandas as pd
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
```

feature extraction 값을 csv로 export하면 결과는 다음과 같이 나온다. (테스트 코드 일부)

![feature extraction result](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F9rXek%2FbtqNqeY63B4%2FoGA4kz5wxwbajsvW2ihTM1%2Fimg.png)

이후 비슷한 음원끼리 묶는 과정은 kmeans clustering으로 진행하였다. 

### KMeans Clustering 

kmeans clustering은 비지도학습으로 이루어지며, 따로 학습 데이터를 확보할 필요없이 매번 실행할 때마다 데이터를 넣어주면 되는데, 앞서 차원을 대폭 줄인 array형태의 데이터값을 input으로 하므로 클러스터링할 때 몇 초면 끝난다. 데이터 간의 상관관계를 기계가 학습할 수 있도록 비지도 학습으로 진행하며 군집에서 중심점(Centroid)을 계속 찾아나가는 KMeans를 이용하였고 이 때 python sklearn 라이브러리를 이용하여 구현하였다.

실제 함수 로직은 다음과 같다.
```
from sklearn.cluster import KMeans
# KMeans Clustering(n_clusters가 군집 개수)
    n_clusters = 10
    model = KMeans(n_clusters)
    model.fit(feat_data)
    labels_data = model.predict(feat_data)
```

### Performance

kmeans에서 중요한 요소 중에 하나가 군집의 개수인데 군집 1개부터 10개까지 성능을 확인할 수 있는 지표인 inertia를 확인해보면 다음과 같다.

![performance](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FYyBGH%2FbtqNx83Wcg1%2FjKahtXcNgiKyz8JAlfGPNK%2Fimg.png)

기울기가 급격한 부분이 가장 적합한 군집 수인데 여기서는 뚜렷한 숫자가 보이지 않는다고 판단했다. 제일 가파른 부분이 2,3 정도 되는데 군집이 2,3개밖에 되지 않으면 유의미한 결과가 나타나지 않아 10개로 설정했다. 그리고 실제로 클러스터링을 돌리면 학습 데이터가 정해져있지 않고 그때그때 들어오는 데이터를 클러스터링하기 때문에 결과가 다소 차이가 나며, 라벨은 0에서 9의 숫자로 랜덤으로 정해진다.

### 가장 유사한 음원 n개 추출하기

같은 군집 내에서 유사한 음원 파일을 찾기 위해 코사인 거리 측정법을 이용했다. 하나의 음원파일 feature 데이터에 20개 요소를 가지기 때문에 이때 한번 더 reshape로 차원을 축소시켰고 출력 형태는 파일명을 key, 유사도 점수(50에서 100점 까지 표현)를 value로 하여 dictionary 타입으로 설정했다.

```
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def cluster_sim(data,idx,n):
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
```

결과 예시는 [('파일명1',99.6),('파일명2',98.1),...]와 같은 형태이다.



### 노래의 진폭을 바탕으로 노래-안무간의 조화도 비교 

노래의 진폭은 python librosa 라이브러리를 사용하였고, 진폭 단위는 데시벨을 기준으로 했다.
librosa에서 측정하는 raw 진폭값은 음수값부터 시작해서 범주가 맞지 않기 때문에 사람이 들을 수 있는 0데시벨부터 다시 데이터 정규화를 진행했다. 그리고 진폭은 구간 전체로 계산하지 않고 박자별로 다시 나누어서 한 구간 안에서도 음악의 진폭(노래 크기)이 어떻게 변하는지 알아볼 수 있도록 하였다.
그래서 코드에서 보면 8박자 기준으로 나누었고 박자의 변화가 8개가 생성되어야 하므로 구간은 9개로 설정해서 9로 나누었다.

그리고 노래-안무간의 조화도를 비교할때 범주형으로 들어가기보다 단계별로 계산하면 더 효율적일 것으로 보아 전체 범위를 1~10단계로 나누어 최종 return 타입이 int 단계값이 되도록 하였다.
진폭 범위는 DB의 노래 진폭 중 최솟값부터 최댓값으로 잡았다.

다음 코드는 amplitude 단계를 출력해주는 코드의 일부분이다.

```
    music, fs = librosa.load(file)
    music_stft = librosa.stft(music)
    music_amp = librosa.amplitude_to_db(abs(music_stft))
    music_fin_amp = [[0 for i in range(len(music_amp[j]))] for j in range(len(music_amp))]

    for i in range(len(music_amp)):
        for j in range(len(music_amp[i])):
            if (music_amp[i][j] >= 0):
                music_fin_amp[i][j] = music_amp[i][j]

    length = len(music_fin_amp)  # dB array 길이: 1025
    len_unit = int(length / 9)  # 8박자 단위로 자르기 때문에 9로 나누기
    amp_list = []
```
