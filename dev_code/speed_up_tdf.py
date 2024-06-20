# 解决两个问题：
#   1. 采用多线程做数据转换
#   2. 用缓冲器装饰 cs_transform 函数进一步减少计算时间，因为它会将 cs_transform函数 所有输入输出缓存，这样遇到同样的输入就不会重复计算
#   3. 重新精简代码，之前佳儒师兄缝合的代码看起来实在有点费劲
import numpy as np
import pandas as pd
import sys,os
import pyproj

"""
@brief 将UTM坐标转换为Carla仿真环境中的经纬度坐标。
        
@param x1 输入的UTM坐标的x值。
@param y1 输入的UTM坐标的y值。
        
@return 转换后的Carla仿真环境的坐标，包含x坐标（纬度）和y坐标（经度取负）。

@details
该函数首先使用预定义的偏移量将输入的UTM坐标还原为真实的UTM坐标，然后
使用tmerc投影坐标系对象将还原后的UTM坐标转换为经纬度坐标。由于Carla使用左手坐标系，
而我们常用的是右手坐标系，因此需要对y坐标取负。
"""
def cs_transform(x1, y1):
    # 地理位置的偏移量，用于还原真实的UTM坐标
    ref_x, ref_y = -40251.76572214719, 326531.9706723457

    # 还原真实的UTM坐标
    x_tmp, y_tmp = x1 - ref_x, y1 - ref_y

    # 定义一个tmerc投影坐标系对象
    # +lat_0 和 +lon_0 应该替换成xodr文件中的经纬度（longitude、latitude）
    p1 = pyproj.Proj("+proj=tmerc +lat_0=39.788261 +lon_0=116.534302 +k=1 +x_0=500000 +y_0=0 +unit=m +type=crs", preserve_units=False)

    # 将真实轨迹的UTM值转换成真实的经纬度
    car_real_lon, car_real_lat = p1(x_tmp, y_tmp, inverse=True)

    # 转换后的坐标
    x_rst = car_real_lat
    y_rst = -car_real_lon  # Carla使用左手坐标系，常用的是右手坐标系，因此y取负
    
    return x_rst, y_rst


def main():
    path2dataset_file = "10039.csv"
    # print(f'dataset path: {path2dataset_file}')
    raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
    colums_we_want = ['timestamp','id','type','x','y'] # 只选择这些节省时间和空间
    raw_data = raw_data[colums_we_want] # 这才是我们确切需要的，不带任何冗余的数据
    # print(raw_data)
    all_ids_that_we_have = raw_data.id.unique()
    # print(all_ids_that_we_have)
    whole_scenario_stamp_min, whole_scenario_stamp_max = min(raw_data.timestamp), max(raw_data.timestamp)
    # print(whole_scenario_stamp_min,whole_scenario_stamp_max) # 1657100477.8  1657100487.7   看起来是10秒的数据

    # during whole simulation
    actor_spawn_time = {}   # 保存了每个ID在仿真开始多久之后 生成 的时间
    actor_destroy_time = {} # 保存了每个ID在仿真开始多久之后 消失 的时间

    # 这一部分for循环中把每一个id的所有行单独列出来，获取其最小和最大时间戳并与整个仿真的起始时间戳求差，（获得这个id的演员
    # 应该在整个仿真的什么时间戳上生成和消失）并按照id标签分别保存在上面定义的两个字典里
    for each_id in all_ids_that_we_have:
        all_row_of_each_id = raw_data.loc[raw_data.id == each_id]
        t_min, t_max = min(all_row_of_each_id.timestamp), max(all_row_of_each_id.timestamp)
        actor_spawn_time[each_id], actor_destroy_time[each_id] = t_min - whole_scenario_stamp_min, \
            t_max - whole_scenario_stamp_min
    
    # 随便选了一个id试验了一下，可以的
    # print(actor_spawn_time[2473222]) # 9.5
    # print(actor_destroy_time[2473222]) # 9.900000095367432

    





if __name__ == '__main__':
    try:
        main()
    
    except Exception as e:
        print(f'main failed: {e}')
    