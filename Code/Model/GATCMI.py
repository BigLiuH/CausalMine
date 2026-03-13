import time

# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import os
os.chdir(os.path.dirname(__file__))
os.environ["PYTORCH_JIT"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 可用于调试
import numpy as np
import time
import pandas as pd
import os
from catboost import CatBoostClassifier
import torch
from matplotlib import pyplot
from numpy import interp

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

import random
from random import randint
import scipy.io

# from keras.layers import merge

# from keras.utils import np_utils, generic_utils

from xgboost import XGBClassifier
# from keras.layers import containers, normalization

from GATpredata import prepare_data

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>这里用的是cbam模块
# from model import GATModel
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>下面用的是eca模块
from GAT import GATModel

from torch import nn, optim
from param2 import parameter_parser

'''
label1是positive样本label
label2是未知样本的label
'''
seed = 468
text = "DF采样"
filename = "评估信息（GAT+CBAM）（DF采样）.txt"

class CMI():
    def __init__(self):
        super().__init__()

        # circ-mi关联矩阵路径
        self.path_interaction = "../../CMI9589dataset/association/ori_circRNAs_miRNAs_association_withoutindex.csv"
        # circ、mi embedding的路径
        self.path_circ_embedding = "../../CMI9589dataset/embedding/DFnode2vec/ciRNAEmbed"  # 用的时候记得加上"mv"
        self.path_mi_embedding = "../../CMI9589dataset/embedding/DFnode2vec/miRNAEmbed"  # 用的时候记得加上"mv"
        
        # 非负样本采样
        # self.path_circ_embedding = "../../CMI9589dataset/embedding/node2vec/ciRNAEmbed"  # 用的时候记得加上"mv"
        # self.path_mi_embedding = "../../CMI9589dataset/embedding/node2vec/miRNAEmbed"  # 用的时候记得加上"mv"
        # 存储核加载分类模块数据的路径
        self.path_cat_model = "../../CMI9589code/DFtest/GATCatModel/MLPmodel_"
        # cat_boost分类模块超参数
        self.cat_epoch = 425  # 代表
        self.cat_depth = 4

        self.threshold = 0.5  # 判断正负样本的阈值常熟
        # circ-mi关联矩阵
        self.interaction = np.loadtxt(self.path_interaction, dtype=float, delimiter=",")

        self.is_loading_embeddings = False  # 判断是否加载了embeddings
        self.is_loading_traindatas = False  # 判断是否加载了训练数据
        # 存储circRNA和miRNA的embedding
        self.circ_embedding = []
        self.mi_embedding = []
        # circRNA和miRNA的数量
        self.circ_number = 2115
        self.mi_number = 821
        # circRNA和miRNA的列表
        self.circ_list = list(np.loadtxt('../../dataop/myCMI/circRNA.txt', dtype=str))
        self.mi_list = list(np.loadtxt('../../dataop/myCMI/miRNA.txt', dtype=str))
        # 二者的序列
        self.mi_seq_list = list(np.loadtxt('../../dataop/myCMI/miRNA_seq.txt', dtype=str))
        self.circ_seq_list = list(np.loadtxt('../../dataop/myCMI/circRNA_seq.txt', dtype=str))

    def load_embbedings(self):
        for i in range(1, 2):
            # 为了从1编号
            self.circ_embedding.append(
                np.loadtxt(self.path_circ_embedding + str(i) + '.csv', dtype=float, delimiter=","))
            self.mi_embedding.append(
                np.loadtxt(self.path_mi_embedding + str(i) + '.csv', dtype=float, delimiter=","))

        for i in range(1, 3):

            print('loading embbedings, mv=', i)
            self.circ_embedding.append(
                np.loadtxt(self.path_circ_embedding + str(i) + '.csv', dtype=float, delimiter=","))
            self.mi_embedding.append(np.loadtxt(self.path_mi_embedding + str(i) + '.csv', dtype=float, delimiter=","))

            print('loading over, mv=', i)
            print()

    def prepare_data3(self):
        if not self.is_loading_embeddings:
            self.load_embbedings()
        self.is_loading_embeddings = True
        # 先把正负样本全部搞出来，然后取全部的正样本，随机取等量的负样本。在此基础上进行训练集和测试集的划分
        # 首先进行测试集和训练集的划分
        n = 2115 * 821
        arr = np.zeros((n, 2), dtype=int)
        cnt = 0
        for i in range(0, 2115):
            for j in range(0, 821):
                arr[cnt][0] = i  # 获取对应circRNA的索引
                arr[cnt][1] = j  # 获取对应miRNA的索引
                cnt += 1
        # 一共有n对，那么我用后面20%作为测试集，前面80%作为训练集

        np.random.shuffle(arr)
        interaction = self.interaction

        # 收集所有正样本
        positive_samples = []# 存储正样本的索引
        for x in range(0, n):
            i = arr[x][0]
            j = arr[x][1]
            if interaction[i][j] == 1:
                positive_samples.append((i, j))

        psnumber = len(positive_samples)
        print(f"正样本数量: {psnumber}")

        # 负样本采样
        negative_samples = []
        with open("negCMI.txt", "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    negative_samples.append((int(parts[0]), int(parts[1])))
        
        print(f"负样本数量: {len(negative_samples)}")
        
        # sys.exit(0)

        # 合并正负样本
        balanced_samples = positive_samples + negative_samples
        np.random.shuffle(balanced_samples)

        # 创建新的样本数组
        tot = len(balanced_samples)
        arr = np.zeros((tot, 2), dtype=int)
        for i, (circ_idx, mi_idx) in enumerate(balanced_samples):
            arr[i][0] = circ_idx
            arr[i][1] = mi_idx

        # 划分训练集和测试集
        train_number = int(tot * 0.8)
        test_number = tot - train_number
        print("#############DEBUGE#############")
        print(f"正样本数: {psnumber}, 负样本数: {len(negative_samples)}, 总样本数: {tot}")
        print(f"训练集大小: {train_number}, 测试集大小: {test_number}")
        print("################################")


        """
        # 统计正负样本数量
        ngnumber = 0
        psnumber = 0
        for x in range(0, n):
            i = arr[x][0]
            j = arr[x][1]  # 获取对应作用对的索引
            if interaction[i][j] == 0:  # 判断是否为负样本
                ngnumber += 1
            elif interaction[i][j] == 1:
                psnumber += 1

        old_arr = arr
        arr = np.zeros((psnumber * 2, 2), dtype=int)  # 创建新数组存储平衡后的样本
        print('len(arr):', len(arr))
        ngnumber = 0
        tot = 0 # 总数
        # 随机选取负样本
        for x in range(0, n):
            i = old_arr[x][0]
            j = old_arr[x][1]
            if interaction[i][j] == 1:
                arr[tot] = old_arr[x]
                tot += 1
            elif interaction[i][j] == 0 and ngnumber < psnumber:
                arr[tot] = old_arr[x]
                tot += 1
                ngnumber += 1

        train_number = int(tot * 0.8)
        test_number = tot - train_number
        print("#############DEBUGE#############")
        print(psnumber, ngnumber, tot, train_number, test_number)
        print("################################")
        """


        np.random.shuffle(arr)  # 打乱顺序
        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        # 下面读取数据,
        for mv in range(1, 3):
            circRNA_fea = self.circ_embedding[mv]
            disease_fea = self.mi_embedding[mv]

            # 下面遍历训练集
            link_number = 0
            train = []  # 存储训练集的特征向量
            testfnl = []
            label1 = []  # 存储训练集样本的标签
            label2 = []
            label22 = []
            ttfnl = []
            for k in range(train_number):
                i = arr[k][0]
                j = arr[k][1]
                if interaction[i, j] == 1:  # for associated
                    label1.append(interaction[i, j])  # label1= labels for association(1)
                    link_number = link_number + 1  # no. of associated samples
                    circRNA_fea_tmp = list(circRNA_fea[i])
                    disease_fea_tmp = list(disease_fea[j])
                    tmp_fea = (circRNA_fea_tmp, disease_fea_tmp)  # concatnated feature vector for an association
                    train.append(tmp_fea)  # train contains feature vectors of all associated samples
                elif interaction[i, j] == 0:  # for no association
                    label1.append(interaction[i, j])  # label2= labels for no association(0)
                    circRNA_fea_tmp1 = list(circRNA_fea[i])
                    disease_fea_tmp1 = list(disease_fea[j])
                    tmp_fea = (
                    circRNA_fea_tmp1, disease_fea_tmp1)  # concatenated feature vector for not having association
                    train.append(tmp_fea)  # testfnl contains feature vectors of all non associated samples

            print("len(train)", len(train))

            train = np.array(train)

            X_train.append(train)  # 存储训练集特征向量
            Y_train.append(label1)  # 存储训练集样本标签
            print('prepare train data over, mv=', mv)
            print(len(train))
            print(len(label1))

            # 下面遍历测试集
            link_number = 0
            train = []
            testfnl = []
            label1 = []
            label2 = []
            label22 = []
            ttfnl = []
            for k in range(train_number, tot):
                i = arr[k][0]
                j = arr[k][1]
                if interaction[i, j] == 1:  # for associated
                    label1.append(interaction[i, j])  # label1= labels for association(1)
                    link_number = link_number + 1  # no. of associated samples
                    # link_position.append([i, j])
                    circRNA_fea_tmp = list(circRNA_fea[i])
                    disease_fea_tmp = list(disease_fea[j])
                    tmp_fea = (circRNA_fea_tmp, disease_fea_tmp)  # concatnated feature vector for an association
                    train.append(tmp_fea)  # train contains feature vectors of all associated samples

                elif interaction[i, j] == 0:  # for no association
                    label1.append(interaction[i, j])  # label2= labels for no association(0)
                    # nonlink_number = nonlink_number + 1
                    # nonLinksPosition.append([i, j])
                    circRNA_fea_tmp1 = list(circRNA_fea[i])
                    disease_fea_tmp1 = list(disease_fea[j])
                    test_fea = (
                    circRNA_fea_tmp1, disease_fea_tmp1)  # concatenated feature vector for not having association
                    train.append(test_fea)  # testfnl contains feature vectors of all non associated samples

            print('prepare test data over, mv=', mv)
            train = np.array(train)

            X_test.append(train)
            Y_test.append(label1)
            print(len(train))
            print(len(label1))

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def calculate_performace(self, test_num, pred_y, labels):  # pred_y = proba, labels = real_labels
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index in range(test_num):
            if labels[index] == 1:
                if labels[index] == pred_y[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if labels[index] == pred_y[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1

        acc = float(tp + tn) / test_num

        if tp == 0 and fp == 0:
            precision = 0
            MCC = 0
            f1_score = 0
            sensitivity = float(tp) / (tp + fn)
            specificity = float(tn) / (tn + fp)
        else:
            precision = float(tp) / (tp + fp)
            sensitivity = float(tp) / (tp + fn)
            specificity = float(tn) / (tn + fp)
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
        print("该测试集数目：",test_num)  # test_num=108
        print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)
        return acc, precision, sensitivity, specificity, MCC, f1_score, tp, tn, fp, fn

    def transfer_array_format(self, data):  # preResult=X  , X= all the miRNA features, disease features
        formated_matrix1 = []
        formated_matrix2 = []
        # 返回分割后的两个数组，分别存储circRNA和miRNA的特征
        for val in data:# data type:<class 'numpy.ndarray'>
            formated_matrix1.append(val[0])  # contains circRNA features
            formated_matrix2.append(val[1])  # contains miRNA features

        return np.array(formated_matrix1), np.array(formated_matrix2)


    class Config(object):
        def __init__(self):
            self.data_path = '../../datasets'
            self.validation = 1
            self.save_path = '../preResult'

            self.epoch = 100# 用不到
            self.alpha = 0.2

    class Sizes(object):
        def __init__(self, dataset):
            self.m = dataset['mm']['preResult'].size(0)
            self.d = dataset['dd']['preResult'].size(0)
            self.fg = 256
            self.fd = 256
            self.k = 32

    def train(self, model, train_data, optimizer, opt):
        model.train()
        # regression_crit = Myloss()
        # one_index = train_data[2][0FS].cuda().t().tolist()
        # zero_index = train_data[2][1].cuda().t().tolist()

        def train_epoch():
            model.zero_grad()
            # 下面这一行出问题了(已解决）
            score, ciRNAEmbed, disEmbed = model(train_data)
            loss = torch.nn.MSELoss(reduction='mean')
            loss = loss(score, train_data['md_p'].cuda())
            # loss = loss(score, train_data['md_p'])
            loss.backward()
            optimizer.step()
            return loss

        def getEmbedding():
            model.zero_grad()
            score, ciRNAEmbed, disEmbed = model(train_data)
            return ciRNAEmbed, disEmbed

        for epoch in range(1, opt.epoch + 1):
            train_reg_loss = train_epoch()
        ciRNAEmbed, disEmbed = getEmbedding()
        print('after model.train()')
        return ciRNAEmbed, disEmbed

    opt = Config()

    def work_on_test_set(self):
        info = f"固定所有随机数({seed})，使用{text}，cat为425,epoch为800（work_on_test_set)nodecirc=0,mi=0"
        if self.is_loading_traindatas == False:
            self.prepare_data3()
        self.is_loading_traindatas = True

        X = self.X_train
        labels = self.Y_train
        X_data1 = [] # 存储训练集miRNA和circRNA的特征
        X_test_data1 = [] # 存储测试集miRNA和circRNA的特征
        y = [] # 存储训练集样本的标签
        y_test = [] # 存储测试集样本的标签
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for j in range(2):
            # 这里的3是指有3个miRNA和circRNA的embedding
            X_data1_, X_data2_ = self.transfer_array_format(
                X[j])  # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)

            X_test_data1_, X_test_data2_ = self.transfer_array_format(self.X_test[j])

            X_data1_ = np.concatenate((X_data1_, X_data2_), axis=1)  # axis=1 , rowwoise concatenation
            X_test_data1_ = np.concatenate((X_test_data1_, X_test_data2_), axis=1)  # axis=1 , rowwoise concatenation

            y_ = np.array(labels[j])
            y_test_ = np.array(self.Y_test[j])
            t = 0
            X_data1.append(X_data1_)
            X_test_data1.append(X_test_data1_)
            y.append(y_)
            y_test.append(y_test_)

        train1 = []
        test1 = []
        train_label = []
        test_label = []
        realLabel = []
        trainLabelNew = []
        probaList = []
        probaCoefList = []

        for i in range(2):
            trainTmp = np.array([x for i, x in enumerate(X_data1[i]) if True])
            testTmp = np.array([x for i, x in enumerate(X_test_data1[i]) if True])
            train_labelTmp = np.array([x for i, x in enumerate(y[i]) if True])
            test_labelTmp = np.array([x for i, x in enumerate(y_test[i]) if True])

            train1.append(trainTmp)
            test1.append(testTmp)
            train_label.append(train_labelTmp)
            test_label.append(test_labelTmp)

        for i in range(2):
            real_labelTmp = []

            for val in test_label[i]:
                if val == 0:  # tuples in array, val[0]- first element of tuple
                    real_labelTmp.append(0)
                else:
                    real_labelTmp.append(1)

            train_label_newTmp = []
            for val in train_label[i]:
                if val == 0:
                    train_label_newTmp.append(0)
                else:
                    train_label_newTmp.append(1)
            class_index = 0
            prefilter_train = train1[i]
            prefilter_test = test1[i]

            # clf = XGBClassifier(n_estimators=self.cat_epoch, max_depth=self.cat_depth)
            clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed, verbose=0,
                                     task_type='GPU')
            clf.fit(prefilter_train, train_label_newTmp)  # ** *Training
            ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # **testing

            clf.save_model(self.path_cat_model + str(i + 1) + '.model')

            proba = self.transfer_label_from_prob(ae_y_pred_prob)
            probaList.append(proba)
            probaCoefList.append(ae_y_pred_prob)
            realLabel.append(real_labelTmp)
            trainLabelNew.append(train_label_newTmp)

        # 计算测试集上的结果
        avgProbCoef = probaCoefList[0]
        for i in range(1, 2):
            tempProb = probaCoefList[i]
            for j in range(len(avgProbCoef)):
                avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
        for i in range(len(avgProbCoef)):
            avgProbCoef[i] = avgProbCoef[i] / 2

        avgProb = self.transfer_label_from_prob(avgProbCoef)
        acc, precision, sensitivity, specificity, MCC, f1_score, tp, tn, fp, fn = self.calculate_performace(
            len(realLabel[1]),
            avgProb,
            realLabel[1])
        # avgProb = transfer_label_from_prob(probaCoefList[9])
        # acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(realLabel[1]), avgProb,
        #                                                                                realLabel[1])

        fpr, tpr, auc_thresholds = roc_curve(realLabel[1], avgProbCoef)
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('raw_DNN', {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score})

        precision1, recall, pr_threshods = precision_recall_curve(realLabel[1], avgProbCoef)

        # pyplot.plot(recall, precision1, label= 'ROC fold %d (AUC = %0.4f)' % (t, auc_score))

        aupr_score = auc(recall, precision1)
        print("testing_set:\n", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
        print('tp=', tp, 'fp=', fp, 'tn=', tn, 'fn=', fn)
        # 测试集结果保存
        metrics_dict = {
            "准确率": acc, "精确率": precision,
            "敏感性": sensitivity, "特异性": specificity,
            "MCC": MCC, "AUC": auc_score,
            "AUPR": aupr_score, "F1分数": f1_score,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }
        # print(metrics_dict)
        self.save_metrics_to_file(metrics_dict, f"{filename}", info)

    def embedding(self):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args = parameter_parser()
        # 以下为GAT的Embedding过程
        print('embedding.......')

        dataset = prepare_data()
        train_data = dataset

        for k in range(1, 3):
            print('k=', k)
            for i in range(self.opt.validation):
                print('-' * 50)
                model = GATModel(args, k)  # parameter_parser()
                model.cuda()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # 下面进行模型训练
                ciRNAEmbed, disEmbed = self.train(model, train_data, optimizer, args)
                print()
            ciRNAEmbed = ciRNAEmbed.detach().cpu().numpy()
            diseaseEmbed = disEmbed.detach().cpu().numpy()

            circPath = self.path_circ_embedding + str(k) + '.csv'
            disPath = self.path_mi_embedding + str(k) + '.csv'  # 其实是miRNA的embedding路径
            np.savetxt(circPath, ciRNAEmbed, delimiter=',')
            np.savetxt(disPath, diseaseEmbed, delimiter=',')
        print('embedding over........')

    def cross_validate(self):
        info = f"固定所有随机数({seed})，使用{text}，cat为425,epoch为800（cross_validate）nodecirc=0,mi=0"
        if self.is_loading_traindatas == False:
            self.prepare_data3()
        self.is_loading_traindatas = True
        X_train = self.X_train
        Y_train = self.Y_train
        Y_test = self.Y_test
        X_test = self.X_test

        X = X_train
        labels = Y_train

        X_data1 = []
        X_data2 = []
        y = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        # ————————————————————————————————————————————————下面这一行关注一下
        num = np.arange(15342)  # 这个是circ_mi关联数量*2，也就是正样本数量*2
        # ————————————————————————————————————————————————上面这一行关注一下！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        np.random.shuffle(num)
        for j in range(2):
            X_data1_, X_data2_ = self.transfer_array_format(
                X[j])  # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)

            X_data1_ = np.concatenate((X_data1_, X_data2_), axis=1)  # axis=1 , rowwoise concatenation
            # 为什么拼接？ # 因为每个circRNA和miRNA都有两个特征向量，一个是miRNA的特征向量，一个是circRNA的特征向量

            X_data1_ = X_data1_[num]  # 代表随机打乱circRNA和miRNA的特征向量

            # X_data2_ = X_data2_[num]# 代表随机打乱circRNA和miRNA的特征向量
            y_ = self.Y_train[j]
            y_ = np.array(y_, dtype=np.int32)
            y_ = y_[num]
            t = 0
            X_data1.append(X_data1_)
            # X_data2.append(X_data2_)
            y.append(y_)

        num_cross_val = 5  # 5折交叉验证

        all_performance = []  # 存储每一折的性能指标

        all_prob = {}
        num_classifier = 3
        all_prob[0] = []
        all_prob[1] = []
        all_prob[2] = []
        all_prob[3] = []
        all_averrage = []

        clf_start_time = time.time()
        for fold in range(num_cross_val):
            # 每一折的3种分类器数据
            train1 = []
            test1 = []
            train_label = []
            test_label = []
            realLabel = []
            trainLabelNew = []
            probaList = []
            probaCoefList = []

            for i in range(2):
                trainTmp = np.array([x for i, x in enumerate(X_data1[i]) if i % num_cross_val != fold])
                testTmp = np.array([x for i, x in enumerate(X_data1[i]) if i % num_cross_val == fold])
                train_labelTmp = np.array([x for i, x in enumerate(y[i]) if i % num_cross_val != fold])
                test_labelTmp = np.array([x for i, x in enumerate(y[i]) if i % num_cross_val == fold])

                # 训练集的特征以及标签
                train1.append(trainTmp)
                test1.append(testTmp)
                # 验证集的特征以及标签
                train_label.append(train_labelTmp)
                test_label.append(test_labelTmp)

            clfName = ''
            # 分类
            # ！！！！！！！！！！分类部分的逻辑：训练9个分类器，然后九个分类器的结果取平均值，作为最后的结果！！！！！！！！！！！！
            # 目前了来看，每个分类器的训练样本不一样啊，因为它的负样本是随机选取的啊。
            for i in range(2):
                real_labelTmp = []
                for val in test_label[i]:
                    if val == 1:
                        real_labelTmp.append(1)
                    else:
                        real_labelTmp.append(0)
                train_label_newTmp = []
                for val in train_label[i]:
                    if val == 1:
                        train_label_newTmp.append(1)
                    else:
                        train_label_newTmp.append(0)
                class_index = 0
                prefilter_train = train1[i]  # 训练集特征数据
                prefilter_test = test1[i]

                # clf = XGBClassifier(n_estimators=self.cat_epoch, max_depth=self.cat_depth)
                clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed,
                                         verbose=0, task_type='GPU')
                clf.fit(prefilter_train, train_label_newTmp)  # ** *Training
                # [:,1] # 取第二列的概率值(正类的概率值)
                ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # **testing

                proba = self.transfer_label_from_prob(ae_y_pred_prob)  #
                probaList.append(proba)
                probaCoefList.append(ae_y_pred_prob)
                realLabel.append(real_labelTmp)  # 加载验证集真实的标签
                trainLabelNew.append(train_label_newTmp)

            # 单独一折求平均
            avgProbCoef = probaCoefList[0]
            for i in range(1, 2):
                tempProb = probaCoefList[i]
                for j in range(len(avgProbCoef)):
                    avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
            for i in range(len(avgProbCoef)):
                avgProbCoef[i] = avgProbCoef[i] / 2

            avgProb = self.transfer_label_from_prob(avgProbCoef)
            acc, precision, sensitivity, specificity, MCC, f1_score, tp, tn, fp, fn = self.calculate_performace(
                len(realLabel[1]), avgProb,
                realLabel[1])

            fpr, tpr, auc_thresholds = roc_curve(realLabel[1], avgProbCoef)
            auc_score = auc(fpr, tpr)
            scipy.io.savemat('raw_DNN', {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score})
            precision1, recall, pr_threshods = precision_recall_curve(realLabel[1], avgProbCoef)

            # pyplot.plot(recall, precision1, label= 'ROC fold %d (AUC = %0.4f)' % (t, auc_score))

            aupr_score = auc(recall, precision1)
            print("AUTO-RF:", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
            all_performance.append(
                [acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score, tp, tn, fp, fn])
            t = t + 1  # AUC fold number

            # pyplot.plot(fpr, tpr, label='%s(AUC = %0.4f)' % (clfName, auc_score))
            pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
            mean_tpr += interp(mean_fpr, fpr, tpr)  # one dimensional interpolation
            mean_tpr[0] = 0.0

            pyplot.xlabel('False positive rate, (1-Specificity)')
            pyplot.ylabel('True positive rate,(Sensitivity)')
            pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
            # pyplot.title('Five classification method comparision')

        clf_time = -clf_start_time + time.time();

        print("clf using ", clf_time, 's')

        mean_tpr /= num_cross_val
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        #
        #
        #
        #
        pyplot.plot(mean_fpr, mean_tpr, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % mean_auc)
        pyplot.legend()

        pyplot.savefig('5-cv_roc.png')

        print('*******AUTO-RF*****')
        print('mean performance of rf using raw feature')
        print(np.mean(np.array(all_performance), axis=0))
        Mean_Result = []
        Mean_Result = np.mean(np.array(all_performance), axis=0)
        Std_Result = []
        Std_Result = np.std(np.array(all_performance), axis=0)

        print('---' * 20)
        print('mean_TP=', Mean_Result[8], 'mean_TN=', Mean_Result[9], 'mean_FP=', Mean_Result[10], 'mean_FN=',
              Mean_Result[11])
        print('Mean-Accuracy=', Mean_Result[0], Std_Result[0], '\n Mean-precision=', Mean_Result[1], Std_Result[1])
        print('Mean-Sensitivity=', Mean_Result[2], Std_Result[2], '\n Mean-Specificity=', Mean_Result[3], Std_Result[3])
        print('Mean-MCC=', Mean_Result[4], Std_Result[4], '\n' 'Mean-auc_score=', Mean_Result[5], Std_Result[5])
        print('Mean-Aupr-score=', Mean_Result[6], Std_Result[6], '\n' 'Mean_F1=', Mean_Result[7], Std_Result[7])
        print('---' * 20)
        # 计算完成后保存结果
        metrics_dict = {
            "cat_epoch": self.cat_epoch, "cat_depth": self.cat_depth,
            "准确率": Mean_Result[0], "准确率标准差": Std_Result[0],
            "精确率": Mean_Result[1], "精确率标准差": Std_Result[1],
            "敏感性": Mean_Result[2], "敏感性标准差": Std_Result[2],
            "特异性": Mean_Result[3], "特异性标准差": Std_Result[3],
            "MCC": Mean_Result[4], "MCC标准差": Std_Result[4],
            "AUC": Mean_Result[5], "AUC标准差": Std_Result[5],
            "AUPR": Mean_Result[6], "AUPR标准差": Std_Result[6],
            "F1分数": Mean_Result[7], "F1分数标准差": Std_Result[7],
            "平均TP": Mean_Result[8], "平均TN": Mean_Result[9],
            "平均FP": Mean_Result[10], "平均FN": Mean_Result[11],
            "交叉验证折数": num_cross_val
        }

        self.save_metrics_to_file(metrics_dict, f"{filename}", info)

    def transfer_label_from_prob(self, proba):
        label = [1 if val >= self.threshold else 0 for val in proba]
        return label

    def save_metrics_to_file(self, metrics_dict, filename, info, mode="a"):

        with open(filename, mode, encoding='utf-8') as f:
            # 写入评估信息
            f.write(f"评估信息：{info}\n")
            # 写入时间戳
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 写入所有指标
            for metric_name, metric_value in metrics_dict.items():
                if isinstance(metric_value, (int, float)):
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {metric_value}\n")

            # 添加分隔线
            f.write("-" * 50 + "\n\n")

    def prediction(self, type="circRNA", name="", sequence=""):
        if not self.is_loading_embeddings:
            self.load_embbedings()
        self.is_loading_embeddings = True

        id = -1
        circ_list = self.circ_list
        mi_list = self.mi_list
        for i in range(len(circ_list)):
            if circ_list[i] == name:
                id = i
                break
        # 如果找得到
        if id != -1:
            probaCoefList = []
            for i in range(2):
                # 先加载catboost模型
                clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed,
                                         verbose=0, task_type='GPU')
                clf.load_model(self.path_cat_model + str(i + 1) + '.model')
                # 然后加载每一对ci-mi组合的特征
                X = []
                circRNA_fea_tmp = list(self.circ_embedding[i][id])
                for j in range(self.mi_number):
                    miRAN_fea_tmp = list(self.mi_embedding[i][j])
                    tmp_fea = (circRNA_fea_tmp, miRAN_fea_tmp)
                    X.append(tmp_fea)

                X = np.array(X)

                X_data1, X_data2 = self.transfer_array_format(X)
                X_data1 = np.concatenate((X_data1, X_data2), axis=1)

                # 进行预测
                ae_y_pred_prob = clf.predict_proba(X_data1)[:, 1]
                probaCoefList.append(ae_y_pred_prob)

            # 下面计算结果
            avgProbCoef = probaCoefList[0]
            for i in range(1, 2):
                tempProb = probaCoefList[i]
                for j in range(len(avgProbCoef)):
                    avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
            for i in range(len(avgProbCoef)):
                avgProbCoef[i] = avgProbCoef[i] / 2
                avgProbCoef[i] = 1.0 / (1 + np.exp(-8 * (avgProbCoef[i] - self.threshold)))

            # 列表
            ans_pred = []  # 一个二元组，第一位是概率，第二位是mi的名字
            for i in range(self.mi_number):
                ans_pred.append((mi_list[i], self.mi_seq_list[i], int(10000 * avgProbCoef[i]) / 10000))

            ans_pred = sorted(ans_pred, key=lambda x: x[2], reverse=True)

            print(ans_pred[:10])
            return ans_pred[:10]

        else:
            id = -1
            circ_list = self.circ_list
            mi_list = self.mi_list
            for i in range(len(mi_list)):
                if mi_list[i] == name:
                    id = i
                    break
            # 如果找不到
            if id == -1:
                print('Can not found this RNA \n')
                return str('Can not find this RNA')

            print('id:', id)

            probaCoefList = []

            for i in range(2):

                # 先加载catboost模型
                clf = CatBoostClassifier(iterations=self.cat_epoch, depth=self.cat_depth, random_seed=seed,
                                         verbose=0, task_type='GPU')
                clf.load_model(self.path_cat_model + str(i + 1) + '.model')
                # 然后加载每一对ci-mi组合的特征
                X = []
                miRNA_fea_tmp = list(self.mi_embedding[i][id])
                for j in range(self.circ_number):
                    circRNA_fea_tmp = list(self.circ_embedding[i][j])
                    tmp_fea = (circRNA_fea_tmp, miRNA_fea_tmp)
                    X.append(tmp_fea)

                X = np.array(X)

                X_data1, X_data2 = self.transfer_array_format(X)
                X_data1 = np.concatenate((X_data1, X_data2), axis=1)

                # 进行预测
                ae_y_pred_prob = clf.predict_proba(X_data1)[:, 1]
                probaCoefList.append(ae_y_pred_prob)

            # 下面计算结果
            avgProbCoef = probaCoefList[0]
            for i in range(1, 2):
                tempProb = probaCoefList[i]
                for j in range(len(avgProbCoef)):
                    avgProbCoef[j] = avgProbCoef[j] + tempProb[j]
            for i in range(len(avgProbCoef)):
                avgProbCoef[i] = avgProbCoef[i] / 2
                avgProbCoef[i] = 1.0 / (1 + np.exp(-8 * (avgProbCoef[i] - self.threshold)))

            # 列表
            ans_pred = []  # 一个二元组，第一位是概率，第二位是mi的名字
            for i in range(self.circ_number):
                ans_pred.append((circ_list[i], self.circ_seq_list[i][:10], int(10000 * avgProbCoef[i]) / 10000))

            ans_pred = sorted(ans_pred, key=lambda x: x[2], reverse=True)

            print(ans_pred[:10])
            return ans_pred[:10]


if __name__ == "__main__":
    torch.set_num_threads(8)
    print(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    cmi = CMI()

    # 先进行embedding操作
    
    # time1 = time.time()
    # cmi.embedding()
    # time2 = time.time()
    # embedding_tiem = time2 - time1
    # minutes = int(embedding_tiem // 60)
    # seconds = embedding_tiem % 60
    # runtime_str = f"embedding程序运行时间：{minutes} 分 {seconds:.2f} 秒\n"
    # with open("runtime_log.txt", "a", encoding="utf-8") as f:
    #     f.write(runtime_str)
    
    # 然后对模型进行五折交叉验证
    time1 = time.time()
    cmi.cross_validate()
    time2 = time.time()
    embedding_tiem = time2 - time1
    minutes = int(embedding_tiem // 60)
    seconds = embedding_tiem % 60
    runtime_str = f"训练程序运行时间：{minutes} 分 {seconds:.2f} 秒\n"
    with open("runtime_log.txt", "a", encoding="utf-8") as f:
        f.write(runtime_str)

    time1 = time.time()
    # 最后再跑测试集
    cmi.work_on_test_set()
    time2 = time.time()
    embedding_tiem = time2 - time1
    minutes = int(embedding_tiem // 60)
    seconds = embedding_tiem % 60
    runtime_str = f"测试程序运行时间：{minutes} 分 {seconds:.2f} 秒\n"
    with open("runtime_log.txt", "a", encoding="utf-8") as f:
        f.write(runtime_str)

    # cmi.prediction(name="hsa_circ_0000615")
    print(1)
    # cmi.prediction(name="hsa_circ_0000615")

    # .prediction(name="miR-338-3p")

# >hsa_circ_0000615
# CAATGATGTTGTCCACTGGGCATGTACTGACCAATGT
# >miR-338-3p
# UCCAGCAUCAGUGAUUUUGUUG
