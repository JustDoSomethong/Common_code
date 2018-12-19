#coding=utf-8
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import argparse
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--Slice',type =str,default='1000')
parser.add_argument('--max_features',type =str,default='1000')
args = parser.parse_args()
###############################################################################
# 处理数据
def data_preprocessing():
    userFeature_data = []
    with open('../data/userFeature.data','r') as f:
        for i,line in enumerate(f):     # 读取文件中所有行
            print line
            line = line.strip().split('|')	# 根据文本特征进行相应分割
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])	# 字符串合并
            userFeature_data.append(userFeature_dict)
        print type(userFeature_data)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature = user_feature[['uid','age','gender','marriageStatus','education','consumptionAbility',
                                     'LBS','interest1','interest2','interest3','interest4','interest5','kw1',
                                     'kw2','kw3','topic1','topic2','topic3','appIdInstall','appIdAction','ct',
                                     'os','carrier','house']]       # 特征显示顺序(pd操作)
        user_feature.to_csv('./userFeature.csv', index=False)       # 保存显示的特征

###############################################################################
# 数据拼接,对不定长特征进行向量特征变换,并保存切片结果
def Data_splicing():
    train = pd.read_csv('./train.csv')		# 读取csv格式文件
    predict = pd.read_csv('./test1.csv')
    ad_feature = pd.read_csv('./adFeature.csv')
    user_feature = pd.read_csv('./userFeature.csv')

    # 将train/test合并,一起编码后分开
    predict['label'] = 0
    data = pd.concat([train, predict])
    data = pd.merge(data,ad_feature,on='aid',how='left')	# 根据公共aid合并特征
    data = pd.merge(data,user_feature,on='uid',how='left')
    data = data.fillna('-1')	# 空值补零

    # features = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
    #             'marriageStatus', 'advertiserId', 'campaignId', 'creativeId','creativeSize','adCategoryId', 'productId',
    #             'productType','appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3',
    #             'interest4', 'interest5','kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId','adCategoryId', 'productId', 'productType']		# 特征维数不大,可以作为one_hot特征
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']	# 特征不定长且维数大,可以作为向量特征

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))      # 对特征进行编码(固定长度)
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    # 向量特征reshape,对不定长特征且维数大特征进行处理
    cv = CountVectorizer(max_features=int(args.max_features))
    for feature in vector_feature:
        cv.fit(data[feature])
        data_trans = cv.transform(data[feature])
        for i in range(len(data_trans.todense())):
            data[feature][i] = ' '.join(map(str,list(data_trans.todense()[i].getA1())))		# 将np.array转换为list后,在将元素转换为字符,最后拼接字符串
    print('cv prepared !')
    # print data_trans.shape

    train = data[data.label != 0]
    test = data[data.label == 0]
    test = test.drop('label', axis=1)

    train = train[['aid','advertiserId','campaignId','creativeId','creativeSize','adCategoryId','productId',
                   'productType','uid','age','gender','marriageStatus','education','consumptionAbility',
                   'LBS','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3',
                   'topic1','topic2','topic3','appIdInstall','appIdAction','ct','os','carrier','house','label']]       # 特征显示顺序(pd操作)

    test = test[['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId',
                   'productType', 'uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility',
                   'LBS', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',
                   'topic1', 'topic2', 'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house',]]      # 特征显示顺序(pd操作),其中test不要label标签

    test.to_csv('./test_res.csv', index=False)  # 保存csv

    # 数据分块Slice,每1000个块保存一个csv文件
    for i in range(len(train)/int(args.Slice)):
        train_res = train.iloc[int(args.Slice)*i:int(args.Slice)*(i+1)]
        # print train_res
        train_res.to_csv('./train_res_'+"%06d"%i+'.csv', index=False)  # 保存csv
    if len(train)%int(args.Slice)!=0:
        train_res = train.iloc[len(train)-len(train)%int(args.Slice):len(train)]
        train_res.to_csv('./train_res_'+"%06d"%(len(train)/int(args.Slice)+1)+'.csv', index=False)  # 保存csv
        # print train_res

if __name__ == '__main__':
    data_preprocessing()
    #Data_splicing()
