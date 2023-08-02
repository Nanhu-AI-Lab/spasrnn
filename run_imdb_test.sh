wget http://www.vuln.cn/wp-content/uploads/2019/08/libstdc.so_.6.0.26.zip
mkdir /share/huawei/srnn/srnn-SRNN_CUDA/libstdc
cd /share/huawei/srnn/srnn-SRNN_CUDA/libstdc
unzip libstdc.so_.6.0.26.zip
cp /libstdc++.so.6.0.26 /usr/lib/x86_64-linux-gnu/
sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.26 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

