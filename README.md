# GCN_predict-Pytorch
Traffic flow predict. Implementation of  graph convolutional network（GCN,GAT,Chebnet） with PyTorch

Requirements：

​	- Pytorch

​	- Numpy

​	- Pandas

​	- Matplotlib



Example Dataset：

​	The datasets are collected by the Caltrans Performance Measurement System (PEMS-04) 

​	Numbers:307 detectors

​	Date:Jan to Feb in 2018 (2018.1.1——2018.2.28)

​	Features:flow, occupy, speed.



Exploring data analysis:

​	1.there is three features:flow,occupy and speed.First, we conduct a visual analysis of data distribution 

​	2.run code: python data_view.py	

​	3.Every node(detector) has three fetures,but two features data distribution are basically stationary, so we only take the first dimension features.

![1606617814776](C:\Users\10189\AppData\Roaming\Typora\typora-user-images\1606617814776.png)



Read dataset:

​	In the traffic_dataset.py file,the get_adjacent_matrix and get_flow_data functions are to read adjacent matrix and flow data.



Model training:

​	In the traffic_preditcion.py,there are three graph convolution neural network models:GCN,ChenNET and GAT.Correspondingly, you only need to modify the 45th line of code in this file, and then observe the different results of model training.

​	python traffic_preditcion.py