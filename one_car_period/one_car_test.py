# 解决两个问题：
#   1. 采用多线程做数据转换
#   2. 用缓冲器装饰 cs_transform 函数进一步减少计算时间，因为它会将 cs_transform函数 所有输入输出缓存，这样遇到同样的输入就不会重复计算
#   3. 重新精简代码
import numpy as np
import pandas as pd
import sys,os
import pyproj

sys.path.append("/home/ubuntu/WCY/carla/PythonAPI/carla/")

import carla
import pymap3d as pm
import math
import random
from agents.tools.misc import draw_waypoints, distance_vehicle
import time
from agents.navigation.controller import VehiclePIDController


class CustomWaypoint:
    """
    @brief 为了适配PythonAPI，包装了一下

    @note 王川页
    @note 联系电话: 13601193495
    @note 邮箱: 553503540@qq.com
    """
    def __init__(self, transform):
        self.transform = transform
        # self.location = location
        # self.rotation = rotation
    # def get_transform(self):
    #     return self.transform


def cs_transform(x1, y1):
    """
    @brief 将UTM坐标转换为Carla仿真环境中的经纬度坐标。
            
    @param x1 输入的UTM坐标的x值。
    @param y1 输入的UTM坐标的y值。
            
    @return 转换后的Carla仿真环境的坐标，包含x坐标（纬度）和y坐标（经度取负）。

    @details
    该函数首先使用预定义的偏移量将输入的UTM坐标还原为真实的UTM坐标，然后
    使用tmerc投影坐标系对象将还原后的UTM坐标转换为经纬度坐标。由于Carla使用左手坐标系，
    而我们常用的是右手坐标系，因此需要对y坐标取负。

    @note 作者: 杨文鲜、钟佳儒、王川页
    @note 注释：王川页
    @note 联系电话: 13601193495
    @note 邮箱: 553503540@qq.com
    """
    # 地理位置的偏移量，用于还原真实的UTM坐标
    ref_x, ref_y = -40251.76572214719, 326531.9706723457

    # 还原真实的UTM坐标
    x_tmp, y_tmp = x1 - ref_x, y1 - ref_y

    # 定义一个tmerc投影坐标系对象
    # +lat_0 和 +lon_0 应该替换成xodr文件中的经纬度（longitude、latitude）
    p1 = pyproj.Proj("+proj=utm +lat_0=0 +lon_0=117 +k=1 +x_0=500000 +y_0=0 +unit=m +type=crs", preserve_units=False)

    # 将真实轨迹的UTM值转换成真实的经纬度
    car_real_lon, car_real_lat = p1(x_tmp, y_tmp, inverse=True)
    car_real_alt = 0

    # 这部分应该用carla世界中原点的真实经纬度代替
    # intersection 地图原点######
    origin_lon = 116.528549
    origin_lat = 39.803799
    ############################

    # <proj value="+proj=tmerc +lat_0=39.802021 +lon_0=116.526939"/>
    # real_9 地图原点############
    # origin_lon = 116.526939
    # origin_lat = 39.802021
    ###########################

    # <proj value="+proj=tmerc +lat_0=39.803612 +lon_0=116.528969"/>
    # one_car_9 地图的原点#######
    # origin_lon = 116.528969
    # origin_lat = 39.803612
    ###########################

    # <proj value="+proj=tmerc +lat_0=39.803242 +lon_0=116.527908"/>
    # one_car_9_2 地图的原点#######
    # origin_lon = 116.527908
    # origin_lat = 39.803242
    #############################

    # <proj value="+proj=tmerc +lat_0=39.802101 +lon_0=116.526299"/>
    # one_car_9_4 地图原点#########
    # origin_lon = 116.526299
    # origin_lat = 39.802101
    #############################

    origin_alt = 0
    # 转换后的坐标
    x_rst, y_rst, z_rst = pm.geodetic2enu(car_real_lat, car_real_lon, car_real_alt, origin_lat, origin_lon, origin_alt)
    y_rst = -y_rst

    # print(f'{x_rst/1000}      {y_rst/1000}')
    return x_rst, y_rst


def df_interpolate(df):
    """
    @brief 这是一个数据预处理函数： 补全缺失帧（线性插值法）
    
    @param df 输入的DataFrame，包含列 id 和 timestamp
    
    @return 插值处理后的DataFrame

    @details
    该函数对输入的 DataFrame 进行处理，确保每个唯一 id 的时间戳序列连续和平滑。如果发现时间戳之间的间隔过大，
    则插入缺失的时间戳，并使用线性插值法填补缺失值，处理后的数据帧将返回给调用者

    @note 注释作者: 王川页
    @note 联系电话: 13601193495
    @note 邮箱: 553503540@qq.com
    """
    ids = df.id.unique()
    interp_df = pd.DataFrame(columns=df.columns)
    for id in ids:
        df_id_veh = df.loc[df.id == id]
        t_min, t_max = min(df_id_veh.timestamp), max(df_id_veh.timestamp)
        if int((t_max - t_min) * 10) > len(df_id_veh):
            df_full = pd.DataFrame(np.linspace(t_min, t_max, num=int((t_max - t_min) * 10)+1)) # 往下三步里必定有排序的代码 
            df_full.columns = ['timestamp']
            df_full.timestamp = df_full.timestamp.map(lambda x: np.round(x, 1))
         
            df_id_veh = pd.merge(df_id_veh, df_full, on='timestamp', how='right')
            df_id_veh.interpolate(method='linear', limit_area='inside', inplace=True)
        interp_df = interp_df.append(df_id_veh, ignore_index=True)

    return interp_df


def main():
    try:
        path2dataset_file = "10039.csv"
        # print(f'dataset path: {path2dataset_file}')
        raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
        colums_we_want = ['timestamp','id','type','x','y', 'theta'] # 只选择这些节省时间和空间
        raw_data = raw_data[colums_we_want] # 这才是我们确切需要的，不带任何冗余的数据
        # print(raw_data)

        ##########################################################################
        # 这一部分代码可以将某一个id的数据单独提取出来，这部分代码用于调试，最后注释掉即可
        #
        some_x = 419518.201555984  # 长弯道的，也是我最倾向于选择的轨迹，只是轨迹点不连续
        # some_x = 419529.565373655  # 长弯道的
        # some_x = 419521.63902373   # 直线的
        # some_x = 419543.576778995  # 短弯道的，第一个数据点就是
        # some_x = 419546.032637883  # 短弯道的



        found = raw_data[raw_data['x'] == some_x]
        print(f'找到符合的一行：\n{found}\n')

        found_id = int(found['id'].values)
        print(f'现在找到该轨迹点属于id为 {found_id} 的对象:\n')
        raw_data = raw_data[raw_data['id'] == found_id]
        print(f'现在将id为 {found_id} 的所有数据提取出来：')
        print(raw_data,'\n')
        ###########################################################################



        print('数据预处理中：')
        processed_data = df_interpolate(raw_data) # 数据预处理，补齐缺失帧
        # processed_data = raw_data # 看看不经过平滑处理的数据什么样子
        # print(processed_data)
        
        all_ids_that_we_have = processed_data.id.unique() # 也就是之前的ids
        # print(all_ids_that_we_have)

        whole_scenario_stamp_min, whole_scenario_stamp_max = min(processed_data.timestamp), max(processed_data.timestamp)
        # print(whole_scenario_stamp_min,whole_scenario_stamp_max) # 1657100477.8  1657100487.7   看起来是10秒的数据

        # during whole simulation
        actor_spawn_time = {}   # 保存了每个ID在仿真开始多久之后 生成 的时间
        actor_destroy_time = {} # 保存了每个ID在仿真开始多久之后 消失 的时间

        # 这一部分for循环中把每一个id的所有行单独列出来，获取其最小和最大时间戳并与整个仿真的起始时间戳求差，（获得这个id的演员
        # 应该在整个仿真的什么时间戳上生成和消失）并按照id标签分别保存在上面定义的两个字典里
        for each_id in all_ids_that_we_have:
            all_row_of_each_id = processed_data.loc[processed_data.id == each_id]
            t_min, t_max = min(all_row_of_each_id.timestamp), max(all_row_of_each_id.timestamp)
            actor_spawn_time[each_id], actor_destroy_time[each_id] = t_min - whole_scenario_stamp_min, \
                t_max - whole_scenario_stamp_min
        
        print('数据预处理完毕！\n')

        print(f'演员-{found_id}生成时间：{actor_spawn_time[found_id]}')
        print(f'演员-{found_id}生成时间：{actor_destroy_time[found_id]}\n')


        # 初始化端口
        try:
            print("初始化Client端口")
            client = carla.Client('127.0.0.1',2000)
            client.set_timeout(5.0)
            print("初始化完成")

        except carla.TimeoutError as T_o:
            print(f'初始化超时：{T_o}')
        

        # 世界配置
        try:
            world = client.get_world()
            settings = world.get_settings()
            
            ################################################################
            # 这部分是同步模式的设置
            #   真正仿真的时候用这一套设置！
            #
            # settings.max_substep_delta_time = 0.01 # 设置最大子步的步长
            # settings.max_substeps  = 10 # 最大子步数，将每一时间步长分解为最多10个子步，具体分解为多少个子步由UE4计算情况动态决定
            # synchronous_master = True
            # settings.synchronous_mode = True 
            # # 因为数据集是10Hz的
            # settings.fixed_delta_seconds = 0.1
            ################################################################

            ################################################################
            # 这一步是一步模式的设置
            # 两种模式的setting不能同时生效
            # 调试的时候用着一套设置，否则没有写word.tick会导致仿真时间飞快
            # 
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None  # 确保不使用固定时间步长
            ################################################################
            
            world.apply_settings(settings) # 使配置生效

        except Exception as e:
            raise RuntimeError('世界信息异常，要么是client.get_world()失败，要么是后续仿真器世界配置出错')
        

        '''
        这部分将所有id的路径点存储在wps_dic字典当中，
        键值为：id、
        数值为：路径点对象组成的数列
        '''
        
        wps_dic = {} # 路径点字典，键值为：id、数值为路径点数列。

        for this_specific_id in all_ids_that_we_have:
            print(f'目前在处理的id是：{this_specific_id}')
            # 将插值处理过后的原始数据集中的所有 “此循环时的id” 的数据打包出来成为一个单独的（暂时的）数据结构
            all_data_of_this_id = processed_data.loc[processed_data.id == this_specific_id]
            # print(all_data_of_this_id)
            wps = [] # 路径点数列

            # if(actor_spawn_time[this_specific_id] == this_frame):
            # 把这个id的数据集转换之后存在 wps 数列当中
            for this_frame in range(len(all_data_of_this_id)): 
                
                x_, y_ = cs_transform(all_data_of_this_id.x.values[this_frame], all_data_of_this_id.y.values[this_frame])

                location = carla.Location(x=x_, y=y_, z=0.2)
                rotation = carla.Rotation(pitch=0.0, yaw=all_data_of_this_id.theta.values[this_frame], roll=0.0)
                # rotation = carla.Rotation(pitch=0.0, yaw=all_data_of_this_id.theta.values[this_frame]-90, roll=0.0) # 不对
                transform = carla.Transform(location, rotation)
                waypoint_wrapped = CustomWaypoint(transform)
                # print(f'打印transform看看是什么类型的：{transform}')

                # wps.append(world.get_map().get_waypoint(transform.location))
                wps.append(waypoint_wrapped)
                # wps.append(waypoint_wrapped)
                

            wps_dic[this_specific_id] = wps
        
        # print(f'看看路径点字典中是什么格式的：{wps_dic}')

        '''
        上一步已经获得了所有这个id对应的轨迹点，接下来debug打印出来看看对不对
        '''
        try:
            # road_way_points = world.get_map().generate_waypoints(distance=1.0)
            # for thispoint in road_way_points:
            #     world.debug.draw_point(thispoint.transform.location,size=0.2,color=carla.Color(g=255),life_time=60)

            draw_waypoints(world, wps_dic[found_id])
            
        except Exception as e:
            print(f'在画waypoint时出错：{e}')
        
        blueprint_library = world.get_blueprint_library()
        cybertruck_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        
        ego_car = world.spawn_actor(cybertruck_bp, wps_dic[found_id][0].transform)
        # 确保车辆保持活跃
        # ego_car.set_autopilot(True)
        # ego_car.set_autopilot(False)


        # 设置PID
        args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
        args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}
                
        PID=VehiclePIDController(ego_car,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)

        target_speed = 50
        next = wps_dic[found_id][1]
        i=1
        # ego_car.set_autopilot(False)
        while True:
            if i == len(wps_dic[found_id])-1:
                    break
            ego_transform = ego_car.get_transform()
            # print(f'ego车辆的transform：{ego_transform}')
            ego_dist = distance_vehicle(next, ego_transform)
            control = PID.run_step(target_speed, next)
            ego_car.apply_control(control)
            if ego_dist < 1.5: 
                i = i + 1
                next = wps_dic[found_id][i]
                control = PID.run_step(target_speed, next)
            
            print(f'目前的场景帧：{i}')
            # world.wait_for_tick()

    finally:
        ego_car.destroy()



if __name__ == '__main__':
    try:
        main()
    
    except Exception as e:
        print(f'main failed: {e}')
    