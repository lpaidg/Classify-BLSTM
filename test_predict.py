#-*- coding:utf-8 -*-
import datetime
import tensorflow as tf
from BLSTM import BLSTM
from util.train_util import get_embedding
from preprocess import get_dict
import sys

# 一些配置和初始化
emb_path = 'wiki1.zh.text.vector'
dict_path = 'index2answer.dict'
gpu_config = "/gpu:0"
num_steps = 100
print ("loading the model,please wait a minute...")
index2answer = get_dict(dict_path)
#index2answer = {1: '采用“流量模块+语音模块+短信模块”自由定制形式，每个模块提供阶梯单价，用户自由定制，任意组合，每月承诺消费不低于19元。', 2: '个人定制套餐为4G套餐，默认开通4G功能。', 3: '若您为个人定制套餐用户，您在手机端进入手机版电信营业厅，完成登录后，点击底部导航“服务”，切换到“办理”页面中进行办理。', 4: '很抱歉，目前只支持新入网用户办理此套餐。', 5: '您可以到本地营业厅办理销户。', 6: '号卡激活后当月月基本功能费和套餐内容均按天折算计扣，按照过渡期资费处理。', 7: '自激活日起，根据订购套餐月基本费按天折算计扣（入网当日到月底），四舍五入到分，套餐内容（如手机上网流量、通话时长、短信条数等）按天数折算，向上取整，业务转换、业务转赠功能次月开放。', 8: '套餐外资费：国内流量0.0001元/KB，国内通话0.15元/分钟，国内短信0.1元/条，其他按标准资费收取。', 9: '具体可变更套餐规则请咨询当地营业厅。', 10: '个人定制套餐暂不支持副卡办理。', 11: ' 个人定制套餐暂不支持办理流量包、语音包、短信包业务。', 12: '个人定制套餐暂不支持流量包业务。', 13: '个人定制套餐暂不支持语音包业务。', 14: '个人定制套餐暂不支持短信包业务。', 15: '转换业务是指仅套餐内订购三种业务量（含套餐内被赠的业务量）可在当月内按照转换规则进行自由互转，套餐外优惠叠加及其他活动转赠或者充值的业务量（例如流量卡充值流量）不在转换范围内，例如：使用剩余语音业务量可按照转换规则转换为流量。转换规则为1分钟语音=2MB流量=2条短信。每月可用于转换的语音上限值为1000分钟（加和值，即多次使用语音进行转换的语音总量不超过1000分钟），可用于转换的流量上限值为1000MB（加和值），可用于转换的短信上限值为100条（加和值）每月最多转换3次，每次转换1种业务。个人定制当月转赠、转换及套餐内剩余流量均可保留到次月，但这些流量在次月不可被再次转赠和转换。套餐业务转换在您账户正常状态下可使用，如您账户存在欠费、停机、挂失等问题，则无法使用该项业务。', 16: '转赠业务是指仅套餐内订购流量的剩余可用量（含套餐内转换业务量，不能二次转赠）可在当月内向同时正在使用本套餐本省的其他用户进行转赠，套餐外优惠叠加或充值的业务量（例如流量卡充值流量）不在转赠范围内。每月可用于转赠的流量上限值为1000MB（加和值，即多次使用流量进行转赠的总量不超过1000MB）每月最多转赠3次，单次转赠1人。每月获赠不受次数限制。个人定制当月转赠、转换及套餐内剩余流量均可保留到次月，但这些流量在次月不可被再次转赠和转换。套餐业务转换在您账户正常状态下可使用，如您账户存在欠费、停机、挂失等问题，则无法使用该项业务。', 17: '您在手机端进入手机版电信营业厅，完成登录后，点击底部导航“服务”，切换到“办理”页面即可进行“套餐变更”办理。', 18: '您在手机端进入手机版电信营业厅，完成登录后，点击底部导航“服务”，切换到“办理”页面即可进行“套餐转换”办理。', 19: '您在手机端进入手机版电信营业厅，完成登录后，点击底部导航“服务”，\t切换到“办理”页面即可进行“套餐转赠”办理。', 20: '您在手机端进入手机版电信营业厅，完成登录后，点击底部导航“服务”，切换到“查询”页面即可查询套餐变更记录。', 21: '您好，有的。当您在网上营业厅使用转赠功能时，可勾选短信提醒被赠人，并写下您对被赠人的留言。当您在手机版电信营业厅使用转赠功能时，会默认给被赠人发送短信提醒，您还可以写下您给被赠人的留言。', 22: '您好，个人定制套餐不激活保留时间因全国各省规则不同，请您咨询本省10000号或到本地营业厅进行咨询。', 23: '1.登录网上营业厅。2.登录手机营业厅客户端。3.联系当地人工客服或者前往营业厅进行查询。', 24: '您可登录网上营业厅http://www.189.cn/ 首页点击费用＞我的话费＞余额查询，即可查询可用余额情况。'}

# 读取词向量
embedding_matrix = get_embedding(emb_path, 'word', 400)

print("building model")
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        initializer = tf.random_uniform_initializer(-0.02, 0.02)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            # 模型实例
            model = BLSTM(400, 20, embedding_matrix, attention=True, num_epochs=100, dropout=0.3, is_training=False)
            saver = tf.train.Saver()
            # 读取训练好的模型参数
            saver.restore(sess, 'models/model')
            # Demo主体部分

            # 预测，string为要预测的字符串
            while True:
                string = str(input("Please input the string:"))
                # 输出预测结果
                label, prob = model.predict_label(sess, string)

                if prob[0][label] > 0.4:
                    print (index2answer[label])
                else:
                    print ("超出电信小客服的了解范围了哦，或许你可以换种方式提问，或者换个问题。")
                #print (index2answer[label])
                '''f2 = open('data/test_data.txt', 'a')
                f2.write(string + "    " + (index2answer[label])+"    "+str(datetime.datetime.now()))
                f2.write("\n")
                f2.close()'''

