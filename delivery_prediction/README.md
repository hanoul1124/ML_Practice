# Delivery Time prediction



- 주어진 `predict_delivery_time.csv` 데이터를 바탕으로, Delivery time을 예측한다.

- 데이터는 다음과 같은 형태로 주어진다

- ```
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Restaurant    11094 non-null  object 
     1   Location      11094 non-null  object 
     2   Cuisines      11094 non-null  object 
     3   AverageCost   11094 non-null  object 
     4   MinimumOrder  11094 non-null  int64  
     5   Rating        9903 non-null   object 
     6   Votes         9020 non-null   float64
     7   Reviews       8782 non-null   float64
     8   DeliveryTime  11094 non-null  int64  
    dtypes: float64(2), int64(2), object(5)
    ```



## 1. Preprocessing

- `Rating` , `Votes`, `Reviews`  feature들의 경우 Null value가 다수 존재한다. 이들을 median value 등으로 임의로 채우기에는 지나치게 많은 양이기 때문에, 학습에 크게 지장을 줄 것이다. 따라서 Null value 데이터들은 삭제해주도록 하였다. 그 결과 8782개의 row로 데이터의 양이 축소되었다. 

- 데이터에서 `Location` 과 `Cuisines`의 경우, 가게의 위치에 대한 텍스트 주소와 가게에서 다루는 메뉴에 대한 카테고리를 복수로 갖는다. 이들 Column에 대해서 학습 가능한 형태로 전처리를 하도록 한다.
    - 1. `Location`
            - `TI College, Law College Road, Pune` 이러한 형태의 string 값이 주어진다. 해당 feature를 수치화시켜 학습이 가능한 형태로 만들기 위해서, 위 주소에 대한 `Longitude`,` Latitude`값을 구하여 새로운 Column으로 추가해주도록 하였다.
            - 해당 동작에는 `geopy` 라이브러리를 사용하였다. 해당 라이브러리에서 `Nominatim` 클래스를 사용해 OpenStreetMap의 API에 위도, 경도 값을 요청하여 새로운 값을 입력했다. Google map이 좀 더 정확한 결과를 돌려주지만, 구글맵 API를 사용하는 과정에서 결제 문제가 발생하여 부득이하게 OSM API를 사용하는 것으로 전환하였다. 이 과정에서 일부 위도/경도값을 찾지 못한 주소들이 존재하여, 일부 데이터가 다시 한 번 삭제되었다. 결과적으로 전체 데이터의 수는 8357개가 되었다.
        2. `Cuisines`
            - `Fast Food, Rolls, Burger, Salad, Wraps` 이러한 형태의 String값이 주어진다. 해당 feature를 다루는 방법으로는 우선 가장 간단하게는 OnehotEncoding 방식을 떠올릴 수 있다. 그러나 전체 카테고리의 수가 중복없이 2000여개 이상이 존재하기 때문에, 지나치게 비효율적인 방식이므로 다른 방법을 찾아야한다. 
            - 이에 대해서 scikit-learn의 `CountVectorizer`를 사용하기로 하였다. 해당 변환기는 문서를 토큰 리스트로 변환하고, 각 문서에서 토큰의 출현 빈도를 센다. 이후 각 문서를 BOW 인코딩 벡터로 변환한다. 즉, 위의 string값에서 각 메뉴 카테고리를 개별적으로 분리한 후에, 모든 메뉴에 대해서 출현 빈도를 표시한 벡터로 변환시켜준다는 것을 의미한다.
            - 이에 위 String값을 `fast_food rolls burger salad wraps`와 같이 replace하여 공백을 기준으로 분리한 후, 각 메뉴 카테고리에 대해서 CountVector를 찾아준다.
            - 해당 CountVector를 각각 새로운 Column으로 데이터에 추가하면, 전체 107개의 Column을 가진 데이터셋이 된다.

- 다음으로 numerical한 value에 대해서 Scale을 조정해준다. scikit-learn에서 제공하는 `StandardScaler`를 사용해 표준 정규분포에 따른 수치 조정을 진행하였다.



## 2. Training

- 전처리가 완료된 데이터를 8:2의 비율로 Train set과 Test set으로 나누었다. 각각 6685개, 1672개의 row를 가진 데이터셋이 되었다.

- 총 3개의 ML Model을 사용하여 학습 정확도를 측정하였다.

    - Linear Regression
    - Random Forest
    - Extra Trees

- 이에 대해 다음의 평가지표를 사용하였다.

    - RMES
    - MAE
    - Under-prediction

- Random Forest와 Extra Trees에 대해서는 하이퍼파라미터의 조정을 위해서 Cross-validation 기법을 사용하고, 이를 scikit-learn의 `GridSearchCV`를 사용해 좀 더 정확한 결과를 도출하는 값을 설정하였다.

    

- 1. `LinearRegression`

        - 우선은 단순한 `LinearRegression`을 수행하여 선형 방정식의 계수를 확인한 결과, 위에서 `Cuisines`를 분리한 각각의 메뉴 카테고리들이 가장 영향력이 크게 작용하는 요소가 되어있음을 확인할 수 있었다.

        ```
        (0.8081401215603721, 'middle_eastern'),
        (0.6211128311267659, 'indian'),
        (0.6004700421878416, 'bakery'),
        (0.5881594896904694, 'kashmiri'),
        (0.5880233476731355, 'konkan'),
        (0.5801136604439029, 'brazilian'),
        (0.43271663689456946, 'naga'),
        (0.4221731043727851, 'mishti'),
        (0.41980621729811485, 'wraps'),
        (0.4189690661237363, 'japanese'),
        (0.4025968221467055, 'Votes'),
         ...
        ```

        - 해당 모델은 다음과 같은 평가 지표가 나타났다.

        - ```
            RMSE: 0.9212939208817778
            MAE: 0.6839482862574592
            UNDER_PRED: 0.35991024682124156
            ```

- 2. `RandomForestRegressor`

        - https://wyatt37.tistory.com/7

        - https://hleecaster.com/ml-random-forest-concept/

        - Tree based bethod는 데이터를 어떠한 규칙에 의해서 연속적으로 분기를 나누어가며 탐색/예측하는 것이다. 어떤 feature에 대해 어떤 값으로 split을 하는지가 중요하다. 즉, Tree method는 feature space에 대해서 겹쳐지지 않는 N개의 공간으로 전체 space를 계층적 분리를 실시하는 것이라고 할 수 있다.

        - 학습은 각 분리된 공간에서의 LossFunction을 minimize하는 방향으로 Split point를 잡도록 한다. 이들 Tree method는 학습 데이터에 오버피팅하는 경향이 있으므로, 가지치기 등을 통해 부작용을 최소화해야 한다.

        - RandomForest는 다수의 의사결정트리로부터 나온 분류 결과를 취합하여 결론을 얻는다. 다수의 트리를 통해 예측하기 때문에, 개별 트리가 오버피팅이 나타난다고 해도 결과에 끼치는 영향력이 줄어들게 되므로, 좀 더 좋은 일반화 성능을 보인다.

        - 랜덤포레스트는 Bagging이라는 프로세스를 통해 트리를 구성하는데, 이는 트리를 구성할 때 사용되는 Feature를 selecion하는 데에 사용된다. 즉, 전체 feature 중 일부 feature만을 사용해 하나의 트리를 구성할 것이며, 이들 feature는 중복 선택이 허용된다는 점을 기억해야 한다. 보통 전체 속성 개수의 제곱근만큼 선택한다. (25개의 속성이 있다면 5개를 선택해 하나의 트리 구성)

        - 해당 모델은 다음과 같은 평가 지표가 나타났다.

        - ```
            RMSE: 0.2843294970711888
            MAE: 0.18044614607612627
            UNDER_PRED: 0.3081525804038893
            ```



- 3. `ExtraTreesRegressor`

        - https://wyatt37.tistory.com/6

        - 엑스트라 트리는 랜덤 포레스트를 기반으로 일부 변화를 준 회귀 모델이다. 좀 더 극단적으로 Random함을 추구하는 모델로, 랜덤포레스트와는 두 가지의 차이점이 있다. 하나는 부트스트래핑의 유무이며, 다른 하나는 Split 선택 기준이다.

        - 랜덤포레스트가 Bootstraping(복원추출-중복허용)을 기반으로 Weak Tree를 생성하는것과 달리, 전체 origin data를 그대로 가져다 사용한다. 즉, 비복원 추출을 사용하는 방식이다. 랜덤포레스트는 주어진 샘플에 대해서 모든 변수(Feature)에 대한 정보 이득(Information Gain)을 계산하고 가장 설명력이 높은 변수의 파티션을 Split node로 선택한다. 그러나 Extra Trees는 무작위로 Feature를 선택하여 그 Feature에 대해서만 파티션을 찾아 Node를 분할한다. 이러한 변화를 통해 랜덤포레스트에 비해서 Bias와 Vairance를 줄일 수 있으며, 나아가 연산속도를 약 1/3으로 줄일 수 있다고 한다.

        - 해당 모델은 다음과 같은 평가 지표가 나타났다.

        - ```
            RMSE: 0.05969480717908102
            MAE: 0.02585310888241018
            UNDER_PRED: 0.23679880329094988
            ```



## 3. Test

- Linear Regression모델을 제외한 두 모델에 대해서만 테스트 성능을 측정했다. 각 모델에 대해서 다음과 같은 테스트 결과가 나타났다.

- ```
    # Random Forest
    RMSE: 0.7710933288847798
    MAE: 0.5041438869863142
    UNDER_PRED: 0.3133971291866029
    
    # Extra Trees
    RMSE: 0.7917964875152023
    MAE: 0.46869412431663876
    UNDER_PRED: 0.29605263157894735
    ```

- Extra Trees가 비교적 더 좋은 성능을 나타내는 것을 확인할 수 있었다.



## 4. PCA

- https://velog.io/@sset2323/06-02.-PCAPrincipal-Component-Analysis
- https://bskyvision.com/347

- 가장 대표적인 차원 축소 방법으로, PCA는 여러 변수 간에 존재하는 상관관계르 이용하여 이 변수들의 상관관계를 대표하는 주성분을 추출해 차원을 축소하는 방법이다. 유실되는 정보를 최소화하면서 차원을 축소할 수 있는 기법. 데이터의 분포에 대한 정보를 파악하는 것으로, 가장 높은 분산을 가지는 데이터의 축을 찾아 이 축이 차원을 축소하는 주성분이 된다. 즉, 분산이 데이터의 특성을 가장 잘 나타내는 것으로 간주한다.

- PCA는 상관계수 행렬(공분산행렬)에 대해서 고유값 분해를 진행하는데, 이 때 고유값의 크기에 따라서 고유 벡터 방향으로 벡터 공간이 늘려지는 것이기 때문에 중요도와 공헌도가 높다고 평가한다. 이를 설명력이 높다고 판단.

- 가장 큰 변동성(분산)을 갖는 첫번째 벡터 축을 생성하고, 이에 직각이 되는 직교 벡터를 두번째 축으로 한다. 그리고 세번째 축은 첫번째와 두번째 축에 직각이 되는 벡터를 축으로 설정한다. 이렇게 생성한 벡터 축에 원본 데이터를 투영하면 벡터 축 개수만큼의 차원으로 차원 축소가 이루어진다. 예를 들어, 위와 같이 3개의 주성분 벡터 축을 찾았다고 하자. 이 때 누적 공헌도를 판단했을 때, 세번째 주성분 벡터는 제외되었고, 남은 2개의 벡터에 원본 데이터를 투영시킨다. 그러면 우리는 2차원으로 축소된 데이터를 얻을 수 있게되는 것이다.

- 전체 데이터의 차원수가 지나치게 많다고 판단, PCA 기법을 적용해보기로 하였다. 지정된 차원수에 따라 PCA를 주성분들이 얼마나 데이터를 잘 표현하는지를 확인해본 결과, 30개 이상의 차원수가 지정될 때 어느정도 효율적인 차원축소 결과가 나타난다고 판단하였다. 이에 PCA로부터 30개의 주성분을 설정하여 차원을 축소하여 이에 대해 위 2, 3번 항목에서 진행한 Train, Test과정을 진행했다.

- PCA가 진행된 데이터셋에 대해서 다음 결과가 도출되었다.

- ```
    # Random Forest Train
    MSE: 0.3066811643388974
    MAE: 0.19481457933489624
    UNDER_PRED: 0.3691847419596111
    
    # Random Forest Test
    RMSE: 0.9797006678044488
    MAE: 0.7877319114156063
    UNDER_PRED: 0.3229665071770335
    
    # Extra Trees Train
    RMSE: 0.25665547350179874
    MAE: 0.16944779532995127
    UNDER_PRED: 0.35706806282722514
    
    # Extra Trees Test
    RMSE: 0.96402941159623
    MAE: 0.7477760641596364
    UNDER_PRED: 0.3253588516746411
    ```

- PCA를 진행한 후의 학습 및 테스트 결과가 이전의 결과에 비해서 상대적으로 부정확하게 나타났다. 이는 차원을 축소하는 과정에서 발생하는 정보의 손실에 의한 것이라 해석할 수 있을 것이다.

