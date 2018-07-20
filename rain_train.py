# coding=utf-8

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
import datetime
import pickle
import time
import os

# PRS_Avg TEM_Avg WIN_S_2mi_Avg Lat Lon Alti dow doy day month t_w24 t_w48 tdif2 t_t24 t_t48 tdif3
# 平均气压 平均气温 平均风速 纬度 经度 海拔 星期 一年的第几天 一月的第几天 月份 昨天的风速 前天的风速 今天与昨天的风速差 昨天的气温 前天的气温 气温差
# scale_max = np.array([1300, 50, 50, 53.55, 135.05, 4000, 6, 366, 31, 12, 50, 50, 50, 50, 50, 50])  # 各要素的最大值
scale_max = np.array([50, 53.55, 135.05, 4000, 6, 366, 31, 12, 50, 50, 50, 50, 50])  # 各要素的最大值
# scale_min = np.array([500, -50, 0, 3.86, 73.66, -10, 0, 1, 1, 1, 0, 0, 0, -50, -50, -50])  # 各要素的最小值
scale_min = np.array([-50, 3.86, 73.66, -10, 0, 1, 1, 1, 0, 0, -50, -50, -50])  # 各要素的最小值
scale_ = scale_max - scale_min


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def get_onehot(y):
    y_1 = list()
    for i in y:
        if i < 10.:
            y_1.append(0)
        # elif i < 25.:
        #     y_1.append(1)
        # elif i < 50.:
        #     y_1.append(2)
        # elif i < 100.:
        #     y_1.append(3)
        # elif i < 250.:
        #     y_1.append(4)
        else:
            y_1.append(1)
    return np.array(y_1)


def set_gpu(gpu_memory_frac=0.2):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac    # 不全部占满显存, 按给定值分配
    # config.gpu_options.allow_growth=True   # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    K.set_session(sess)


def lr_schedule(epoch):
    lr = 1e-1
    if epoch > 1500:
        lr = 1e-4
    elif epoch > 1000:
        lr = 1e-3
    elif epoch > 500:
        lr = 1e-2
    print 'Learning rate:', lr
    return lr


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


if __name__ == '__main__':
    features = [
        # 'PRS_Avg',            # 平均气压
        'TEM_Avg',            # 平均气温
        # 'WIN_S_2mi_Avg',      # 平均风速
        'Lat',                # 纬度
        'Lon',                # 经度
        'Alti',               # 海拔
        'dow',                # 星期
        'doy',                # 一年的第几天
        'day',                # 一月的第几天
        'month',              # 月份
        't_w24',              # 昨天的风速
        't_w48',              # 前天的风速
        # 'tdif2',              # 今天与昨天的风速差
        't_t24',              # 昨天的气温
        't_t48',              # 前天的气温
        'tdif3'               # 今天与昨天的气温差
    ]

    base_dir = "/home/sinoeco/hdd/pzw/data/"
    a = np.random.randint(1, 10)
    # a = 1
    print "随机文件是%s" % a
    df1 = pd.read_csv(os.path.join(base_dir, "all_w5_01", str(a) + ".csv"))  # 无雨的数据
    # print df1["Lat"].unique().shape
    # print df1["Lon"].unique().shape
    # print df1["Alti"].unique().shape
    df2 = pd.read_csv(os.path.join(base_dir, "all_w5_02.csv"))  # 小雨及以上的数据
    # from preprocess_2 import preprocess
    # df = preprocess(df)
    # print df.head()
    x1 = df1[features].values
    x2 = df2[features].values
    y1 = df1['PRE_Time_2020'].values
    y2 = df2['PRE_Time_2020'].values
    print "无雨的数据", x1.shape, y1.shape
    print "有雨的数据", x2.shape, y2.shape

    # 合并两部分的数据
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    print "所有天气的数据", x.shape, y.shape

    y = get_onehot(y)
    plt.hist(y)
    plt.xlabel('rain level')
    plt.ylabel('rain days')
    plt.show()

    x = np.array([np.clip(xi, a_min=scale_min, a_max=scale_max) for xi in x])  # 异常值处理
    # x = np.array([(xi - scale_min) / scale_ for xi in x])  # 数据归一化处理

    random_index = np.arange(x.shape[0])
    np.random.shuffle(random_index)
    x = x[random_index]
    y = y[random_index]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print "train data shape:", x_train.shape, "test data shape:", x_test.shape

    if 0:
        np.set_printoptions(linewidth=200)
        print np.max(x, axis=0)
        print np.min(x, axis=0)

    if 0:
        from sklearn.svm import SVC

        clf = SVC(kernel='linear')
        # clf = SVC(C=100.0, kernel='rbf', gamma=0.01)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        num = 0
        preds = preds.tolist()
        for i, pred in enumerate(preds):
            if int(pred) == int(y_test[i]):
                num += 1
        print "clf score", float(num) / len(preds)

        clf2 = SVC(C=100.0, kernel='rbf', gamma=0.01)
        clf2.fit(x_train, y_train)

        preds2 = clf2.predict(x_test)
        num2 = 0
        preds2 = preds2.tolist()
        for i, pred in enumerate(preds2):
            if int(pred) == int(y_test[i]):
                num2 += 1
        print "clf2 score", float(num2) / len(preds2)

    if 0:
        from keras.utils import to_categorical
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization, Activation
        from keras.optimizers import SGD
        from keras.callbacks import TensorBoard, LearningRateScheduler

        mlp_model = Sequential()
        # mlp_model.add(Dense(128, activation='relu', input_dim=x.shape[1]))
        mlp_model.add(Dense(128, input_dim=x.shape[1]))
        mlp_model.add(BatchNormalization())
        mlp_model.add(Activation('relu'))
        mlp_model.add(Dense(64))
        mlp_model.add(BatchNormalization())
        mlp_model.add(Activation('relu'))
        mlp_model.add(Dense(4, activation='softmax'))
        mlp_model.compile(optimizer=SGD(lr=lr_schedule(0), momentum=0.9, decay=0.05, nesterov=True),
                          loss='categorical_crossentropy', metrics=['accuracy'])
        print "网络结构是\n", mlp_model.summary()

        set_gpu(0.4)

        y = to_categorical(y, num_classes=4)

        batch_size = 32768
        epochs = 2000
        suffix = 'v06'

        tblogger = TensorBoard(log_dir='tblogger_'+str(suffix), histogram_freq=0, write_graph=True, write_images=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        mlp_model.fit(
            x=x, y=y,
            batch_size=batch_size, epochs=epochs,
            validation_split=0.1,
            callbacks=[tblogger, lr_scheduler]
        )
        mlp_model.save("b"+str(batch_size)+"e"+str(epochs)+str(suffix)+".h5")

    if 1:
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB

        if 0:
            models = [("xgb", XGBClassifier()),
                      ("LogisticReg", LogisticRegression()),
                      ("RandomFo", RandomForestClassifier()),
                      ("DecisionT", DecisionTreeClassifier()),
                      ("AdaBoost", AdaBoostClassifier()),
                      ("Naive_bayes", GaussianNB())]
            for name, model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                predictions = [round(value) for value in y_pred]
                acc = accuracy_score(y_test, predictions)
                print name, "accuracy is %.2f%%"%(acc * 100.)

        save_dir = "model_cl2/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print "current time is ", current_time()
        model01 = XGBClassifier()
        model02 = LogisticRegression()
        model03 = RandomForestClassifier()
        model04 = DecisionTreeClassifier()
        model05 = AdaBoostClassifier()
        model06 = GaussianNB()

        print "current time is ", current_time()
        model01.fit(x_train, y_train)
        with open(save_dir+"model01.pickle", "wb") as fw:
            pickle.dump(model01, fw)
        # y_pred = model01.predict(x_test)
        y_pred_prob01 = model01.predict_proba(x_test)
        # predictions = [round(value) for value in y_pred]
        acc = accuracy_score(y_test, np.argmax(y_pred_prob01, axis=1))
        print "xgb", "accuracy is %.2f%%" % (acc * 100.)

        print "current time is ", current_time()
        model02.fit(x_train, y_train)
        with open(save_dir+"model02.pickle", "wb") as fw:
            pickle.dump(model02, fw)
        # y_pred = model02.predict(x_test)
        y_pred_prob02 = model02.predict_proba(x_test)
        # predictions = [round(value) for value in y_pred]
        acc = accuracy_score(y_test, np.argmax(y_pred_prob02, axis=1))
        print "LogisticReg", "accuracy is %.2f%%" % (acc * 100.)

        print "current time is ", current_time()
        model03.fit(x_train, y_train)
        with open(save_dir+"model03.pickle", "wb") as fw:
            pickle.dump(model03, fw)
        # y_pred = model03.predict(x_test)
        y_pred_prob03 = model03.predict_proba(x_test)
        # predictions = [round(value) for value in y_pred]
        acc = accuracy_score(y_test, np.argmax(y_pred_prob03, axis=1))
        print "RandomFo", "accuracy is %.2f%%" % (acc * 100.)

        print "current time is ", current_time()
        model04.fit(x_train, y_train)
        with open(save_dir+"model04.pickle", "wb") as fw:
            pickle.dump(model04, fw)
        # y_pred = model04.predict(x_test)
        y_pred_prob04 = model04.predict_proba(x_test)
        # predictions = [round(value) for value in y_pred]
        acc = accuracy_score(y_test, np.argmax(y_pred_prob04, axis=1))
        print "DecisionT", "accuracy is %.2f%%" % (acc * 100.)

        print "current time is ", current_time()
        model05.fit(x_train, y_train)
        with open(save_dir+"model05.pickle", "wb") as fw:
            pickle.dump(model05, fw)
        # y_pred = model05.predict(x_test)
        y_pred_prob05 = model05.predict_proba(x_test)
        # predictions = [round(value) for value in y_pred]
        acc = accuracy_score(y_test, np.argmax(y_pred_prob05, axis=1))
        print "AdaBoost", "accuracy is %.2f%%" % (acc * 100.)

        print "current time is ", current_time()
        model06.fit(x_train, y_train)
        with open(save_dir+"model06.pickle", "wb") as fw:
            pickle.dump(model06, fw)
        # y_pred = model06.predict(x_test)
        y_pred_prob06 = model06.predict_proba(x_test)
        # predictions = [round(value) for value in y_pred]
        acc = accuracy_score(y_test, np.argmax(y_pred_prob06, axis=1))
        print "Naive_bayes", "accuracy is %.2f%%" % (acc * 100.)

        print "current time is ", current_time()
        # ==========================
        # **********模型集成**********
        # ==========================
        ensemble1 = (y_pred_prob01 + y_pred_prob02 + y_pred_prob03 +
                     y_pred_prob04 + y_pred_prob05) / 5.
        preds1 = np.argmax(ensemble1, axis=1)
        acc1 = accuracy_score(y_test, preds1)
        print "ensemble1", "accuracy is %.2f%%" % (acc1 * 100.)
        print "current time is ", current_time()

        ensemble2 = (y_pred_prob01 + y_pred_prob02 + y_pred_prob03 +
                     y_pred_prob04 + y_pred_prob05 + y_pred_prob06) / 6.
        preds2 = np.argmax(ensemble2, axis=1)
        acc2 = accuracy_score(y_test, preds2)
        print "ensemble2", "accuracy is %.2f%%" % (acc2 * 100.)
        print "current time is ", current_time()

        ensemble3 = (y_pred_prob01 + y_pred_prob03 + y_pred_prob05) / 3.
        preds3 = np.argmax(ensemble3, axis=1)
        acc3 = accuracy_score(y_test, preds3)
        print "ensemble3", "accuracy is %.2f%%" % (acc3 * 100.)
        print "current time is ", current_time()

