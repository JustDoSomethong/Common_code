从caffe/tools/extra拷贝文件：parse_log.py，extract_seconds.py，parse_log.sh，plot_training_log.py.example，并将 plot_training_log.py.example 改为 plot_training_log.py
终端中输入： python parse_log.py out.log ./
此时，会出现out.log.train 和 out.log.test 两个文件，里面保存了提取出来的 seconds、loss、accuracy 信息
绘制accuracy和loss曲线,终端中输入：python plot_training_log.py 6 trainloss.png out.log
同样的./test_plot.py为自带的画图程序

