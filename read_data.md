 # 建筑楼层大规模能量预测代码笔记
 记录的都是一些python的写法问题，归结于自己太菜了，python如此强大的功能，很多都没看到过
	
	from sklearn.preprocessing import LabelEncoder
	from sklearn.model_selection import KFold
	sklearn是大家共同开发维护的一个非常方便的机器学习库，得到大家的认可，功能十分强大，还是得慢慢的去了解并使用它啊
**关于LabelEncoder**
sklearn的preprocessing包主要是用于数据预处理的。
preprocessing中的LabelEncoder用于将标签进行标准化，它可以将标签分配为0-n_classes-1之间的编码
后面再说本次代码用到的地方



	train_df = pd.read_csv('xxx.csv')
	train_df = train_df.query('not (building_id <= 104 & meter ==0 & timestamp <="2016-05-20")')
	利用pandas读取csv文件，返回的为dataFrame对象，可以调用head(),describe(),info()等各种方法来查看数据的一些情况
	后面一句代码可以记住，有点类似于数据库语句的感觉，通过query（）将数据块中符合要求的取出。
	还有一种表达就是   train_df[train_df.column2 > train_df.colume4]类似于这种表达方式。

```
 	time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]
```
这里又是一个盲区，因为训练数据中的timestamp列下的数据都是字符串，所以我们需要把他们转换成标准时间类型，这里就用到strptime()
Python格式化日期时间的函数为datetime.datetime.strftime()；由字符串转为日期型的函数为：datetime.datetime.strptime()，两个函数都涉及日期时间的格式化字符串，列举如下:

ebay中时间格式为‘Sep-21-09 16:34’
则通过下面代码将这个字符串转换成datetime

>c = datetime.datetime.strptime('Sep-21-09 16:34','%b-%d-%y %H:%M');
> c
datetime.datetime(2009, 9, 21, 16, 34)
又如：datetime转换成字符串
> datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S');
'Sep-22-09 16:48:08'

大致可以理解了，strptime()函数，第一个参数为要转换的字符串，第二个参数是准备要转换成的time格式。
然后代码接下来计算了一下数据最早时间和最晚时间的时间差，这里再罗列出datatime.seconds()和datatime.total_seconds()之间的区别

其实seconds获取的是仅仅是时间差的秒数，忽略微秒数，忽略天数。
total_seconds()是获取两个时间之间的总差。

>import datetime   
> t1 = datetime.datetime.strptime("2017-9-06 10:30:00", "%Y-%m-%d %H:%M:%S")
>  t2 =datetime.datetime.strptime("2017-9-06 12:30:00", "%Y-%m-%d %H:%M:%S") 
> interval_time = (t2 - t1).seconds  
> 输入的结果：7200 
> total_interval_time =(t2 - t1).total_seconds() 
> 输出结果是: 7200.0 
> print interval_time 
> print total_interval_time  
> 
> 换一个方式 
> t1 = datetime.datetime.strptime("2017-9-06 10:30:00", "%Y-%m-%d %H:%M:%S") 
> t2 = >datetime.datetime.strptime("2017-9-08 12:30:00", "%Y-%m-%d %H:%M:%S") 
> interval_time = (t2 - t1).seconds  
> 输入的结果：7200
> total_interval_time = (t2 - t1).total_seconds() 
> 输出结果是:180000.0 
> print interval_time 
> print total_interval_time 
> td = (t2 - t1) print((td.microseconds + (td.seconds + td.days * 24 * 3600) * >10**6) /10**6)  

所以根据实际情况还是选择datatime.total_seconds()

	#datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
	td = datetime.timedelta(6, 5, 1, 800, 12, 3) 
	print td      # 6 days, 3:12:05.800001
	print td.seconds      # 11525 忽略微秒和天
	print td.total_seconds()     # 529925.800001


numpy中setdiff1d(ar1,ar2,assume_unique=Flase)
1.功能：找到2个数组中集合元素的差异。

2.返回值：在ar1中但不在ar2中的已排序的唯一值。

3.参数：

ar1：array_like 输入数组。
ar2：array_like 输入比较数组。
assume_unique：bool。如果为True，则假定输入数组是唯一的，即可以加快计算速度。 默认值为False。


np.concat
concatenate((a1, a2, …), axis=0) 
数组拼接函数 
参数: 
a1,a2……为要拼接的数组 
axis为在哪个维度上进行拼接，默认为0
> a = np.array([[1, 2], [3, 4]])
> b = np.array([[5, 6]])
>np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
传入的数组必须具有相同的形状，这里的相同的形状可以满足在拼接方向axis轴上数组间的形状一致即可. 
np.concatenate((a, b), axis=1)会报错

代码中用到的是pandas中的concat方法，用于将数据进行融合，有点类似于上面的numpy
这里贴上别人对于pandas 数据合并方法的记录链接
https://blog.csdn.net/stevenkwong/article/details/52528616 讲的很详细

前面说到了特征工程中的LabelEncoder，LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：

fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。 
fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。 
inverse_transform(y)：根据索引值y获得原始数据。 
transform(y) ：将y转变成索引值。
> from sklearn import preprocessing
> le = preprocessing.LabelEncoder()
> le.fit([1, 2, 2, 6])
LabelEncoder()
> le.classes_
array([1, 2, 6])
> le.transform([1, 1, 2, 6]) 
array([0, 0, 1, 2]...)
> le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])

> le = preprocessing.LabelEncoder()
> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
> le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']

```
le = LabelEncoder()
df["primary_use"] = le.fit_transform(df["primary_use"])
代码中将primary_use这一列进行了编码，原来建筑的基本用途大概有一个教育使用、娱乐使用、办公使用等等，LabelEncoder先将他们进行fit即插入操作，然后再转换为索引值，这样就可以用于训练数据了。
```


```
train_df = train_df.merge(buildinf_df,left_on='building_id',right_on='building_id',how='left')
train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
del weather_df
gc.collect()
```
这里介绍了一个dadaframe比较重要的功能，可以将几个dataframe进行可选择的合并操作，这对于数据量较大而且特征维度很多的数据来说是必要的，把他们分开存储，但是当训练数据时还是要合并在一起。
merge的参数

on：列名，join用来对齐的那一列的名字，用到这个参数的时候一定要保证左表和右表用来对齐的那一列都有相同的列名。

left_on：左表对齐的列，可以是列名，也可以是和dataframe同样长度的arrays。

right_on：右表对齐的列，可以是列名，也可以是和dataframe同样长度的arrays。

left_index/ right_index: 如果是True的haunted以index作为对齐的key

how：数据融合的方法
	* inner的话就是两个dataframe公共部分
	* outer的话就是两个dataframe的所有部分
	* left的话就是保留左边dataframe的部分
	* right的话就是保留右边dataframe的部分
	* 未指定how的时候默认为left

sort：根据dataframe合并的keys按字典顺序排序，默认是，如果置false可以提高表现。

暂时先记录这么多，也从中认识到numpy和pandas以及sklearn这几个包的强大以及自己对于他们的陌生，还是得多学习啊，多实践啊


