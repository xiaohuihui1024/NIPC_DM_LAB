import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from my_classifier.knn import KNN
def NSL_preprocessing():
    # 归一化的
    zero_ones = "duration,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate,unknown".split(',')
    # 数值化的
    label_attrs =""
    # 独热编码
    one_hot_attrs = ["protocol_type"]
    # 自定义处理
    # flag：SF=1,其他0
    DIY = ["flag"]
    # 暂不处理 service 70种
    nouse = ["service"]

    df_train = pd.read_csv("datas/NSL-KDD/20 Percent Training Set.csv", low_memory=False)
    df_test = pd.read_csv("./datas/NSL-KDD/Small Training Set.csv", low_memory=False)

    # 得到标注
    train_label = df_train["class"]
    test_label = df_test["class"]
    df_train = df_train.drop(columns=["class"]+nouse)
    df_test = df_test.drop(columns=["class"]+nouse)
    # 归一化
    for feature in zero_ones:
        df_train[feature] = MinMaxScaler().fit_transform(df_train[feature].values.reshape(-1,1)).reshape(1,-1)[0]
        df_test[feature] = MinMaxScaler().fit_transform(df_test[feature].values.reshape(-1,1)).reshape(1,-1)[0]
    # 独热编码
    df_train = pd.get_dummies(df_train, columns=one_hot_attrs)
    df_test = pd.get_dummies(df_test, columns=one_hot_attrs)
    # 独立处理
    df_train["flag"] = [map_flag(s) for s in df_train["flag"].values]
    df_test["flag"] = [map_flag(s) for s in df_test["flag"].values]

    clf = KNN(k=3)
    clf.fit(df_train.values, train_label.values)
    score_train = clf.score()
    print('train accuracy: {:.3}'.format(score_train))

    y_test_pred = clf.predict(df_test.values)
    print('test accuracy: {:.3}'.format(clf.score(test_label.values, y_test_pred)))

def map_flag(f):
    if f == "SF":
        return 1
    else:
        return 0


def main():
    NSL_preprocessing()
    pass

if __name__ == '__main__':
    main()