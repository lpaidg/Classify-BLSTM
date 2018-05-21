#-*- coding:utf-8 -*-
#使用说明：运行时这个test_dataset.py文件和测试数据.txt文件放在同一个目录下，把下边的参数data_path里的内容test_data.txt更改成数据文件名称运行即可。
# 如果最后能输出“success”则数据无格式错误。有格式错误请根据指示的行数更改格式。
data_path = "test_data.txt"

def test_data(path):
    """
    得到数据
    :param path:数据路径
    :return:
    """
    lines = open(path).read().strip().split('\n')
    pairs = [[s for s in l.split('    ')] for l in lines]
    for i in range(len(pairs)):
        try:
            print (pairs[i][1])
        except IndexError:
            print ("第"+str(i)+"行数据格式错误")
        else:
            print ("success")

if __name__ == '__main__':
    test_data(data_path)