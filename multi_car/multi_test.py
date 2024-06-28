'''
待优化：

    1.生成车的location，可以先获取地图的高度z，然后再在这个基础上加0.1，这样所有车都能比较稳妥的生成

    2. 数据处理的多线程优化，我认为，之后这部分可以独立出来成为一个数据预处理脚本，将所有原始数据集的csv都经过转换之后保存在新的csv当中。

    3. 数据处理的多线程优化，现在5辆车就已经很慢了

'''

import numpy as np
import pandas as pd
import sys,os
import pyproj
import pymap3d as pm
import math
import random
import time

sys.path.append("/home/ubuntu/WCY/carla/PythonAPI/carla/")
# sys.path.append("C:/Users/55350/Desktop/WindowsNoEditor/PythonAPI/carla")

import carla
from agents.tools.misc import draw_waypoints, distance_vehicle # type: ignore
from agents.navigation.controller import VehiclePIDController  # type: ignore


def get_z(world,x,y):
    '''
    @brief 此函数用于获取xy精确值的地图z值，用于生成车辆时不发生卡进地面的碰撞错误

    @note 作者：wcy
    '''
    map = world.get_map()
    location = carla.Location(x=x,y=y)
    z_map = map.get_waypoint(location).transform.location.z

    return z_map


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
        path2dataset_file = "../data/10039_3_car.csv"
        # print(f'dataset path: {path2dataset_file}')
        raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
        # print(raw_data)

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
            
        print('数据预处理完毕！\n')  # 这一部分包括了按照时间先后排序
        # print(f'看看生成时间字典：{actor_spawn_time}')   
        # for each in all_ids_that_we_have:
        #     print(f'演员 {each} 的生成时间点：{actor_spawn_time[each]}')

        print(f'开始进行坐标转换：')
        wps_dic = {} # 路径点字典，键值为：id、数值为路径点数列。
        # 才5个车就已经看到了需要多线程改进的影子了--------------------------待修改2
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

                location = carla.Location(x=x_, y=y_, z=0.5)  # z的值，可以优化成，获得生成点地图的高度，然后将z的值就改为地图的高度加0.1 #  还需优化的地方------------待修改1
                # rotation = carla.Rotation(pitch=0.0, yaw=all_data_of_this_id.theta.values[this_frame], roll=0.0)
                rotation = carla.Rotation(pitch=0.0, yaw=-all_data_of_this_id.theta.values[this_frame]+45, roll=0.0) # 不对
                transform = carla.Transform(location, rotation)
                waypoint_wrapped = CustomWaypoint(transform)
                # print(f'打印transform看看是什么类型的：{transform}')

                # wps.append(world.get_map().get_waypoint(transform.location))
                wps.append(waypoint_wrapped)
                # wps.append(waypoint_wrapped)
                    
            wps_dic[this_specific_id] = wps   # 这个当中存的键值是 id， 其value是数列，数列的每个元素是一个 CustomWaypoint实例
            
        # print(f'看看路径点字典中是什么格式的：{wps_dic}')

        '''
        这里也许需要把所有的经过转换过后的数据都存在一个新的csv当中，这样就不用每次relog都重新处理一边数据，这样非常快了
                                待修改3！
        '''
        

        try:
            print("初始化Client端口")
            client = carla.Client('127.0.0.1',2000)
            client.set_timeout(5.0)
            print("初始化完成")

        except carla.TimeoutError as te:
            print(f'初始化超时：{te}')

        try:
            world = client.get_world()
            settings = world.get_settings()
                
            ################################################################
            # 这部分是同步模式的设置
            #   真正仿真的时候用这一套设置！
            #
            settings.max_substep_delta_time = 0.01 # 设置最大子步的步长
            settings.max_substeps  = 10 # 最大子步数，将每一时间步长分解为最多10个子步，具体分解为多少个子步由UE4计算情况动态决定
            # 因为数据集是10Hz的
            settings.fixed_delta_seconds = 0.1
            synchronous_master = True
            settings.synchronous_mode = True 
            
            ################################################################

            ################################################################
            # 这一步是异步模式的设置
            # 两种模式的setting不能同时生效
            # 调试的时候用着一套设置，否则没有写word.tick会导致仿真时间飞快。 这个问题解决了，通过time.sleep来推迟world.tick
            # 
            # settings.synchronous_mode = False
            # settings.fixed_delta_seconds = None  # 确保不使用固定时间步长
            ################################################################
                
            world.apply_settings(settings) # 使配置生效

        except Exception as e:
            raise RuntimeError('世界信息异常，要么是client.get_world()失败，要么是后续仿真器世界配置出错')

        try:
            for each_id in all_ids_that_we_have:
                draw_waypoints(world, wps_dic[each_id])
                
        except Exception as e:
            print(f'在画waypoint时出错：{e}')





        print(f'开始actor一系列初始化')
        blueprint_library = world.get_blueprint_library()
        cybertruck_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        
        chapter_actor_list = [] # 这一段数据集，我管它叫章节，仅仅是因为这么叫好玩，chapter_actor_list包含这一章节所有的演员

        args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
        args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}
        
        # args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
        # args_long_dict = {'K_P': 3,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}


        pid_dict = {}

        target_speed = 29
        

        # for every_car in all_ids_that_we_have:

            # actor_tag = str(every_car) # 将这个actor与id相挂钩，方便之后直接操控相应id的actor
            # cybertruck_bp.set_attribute('role_name',actor_tag)
            # actor = world.spawn_actor(cybertruck_bp, wps_dic[every_car][0].transform)
            # PID=VehiclePIDController(every_car,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
            # next = wps_dic[every_car][1] # 这一行是否有用
            # chapter_actor_list.append(actor)

        # print(f'现在actor的数量是：{len(chapter_actor_list)}')

        

            # for this_actor in chapter_actor_list:
            #     if i == 99:
            #         break
            #     ego_transform = this_actor.get_transform()
            #     # print(f'ego车辆的transform：{ego_transform}')
            #     ego_dist = distance_vehicle(next, ego_transform)
            #     control = PID.run_step(target_speed, next)
            #     this_actor.apply_control(control)
            #     if ego_dist < 1.5: 
            #         i = i + 1
            #         next = wps_dic[this_actor.attributes['role_name']][i]
            #         control = PID.run_step(target_speed, next)

        '''
        接下来需要把所有actor的waypoint的第一个点，也就是生成点的高度，定制化成该位置map的高度，这样会好很多
        '''
        for this_actor in all_ids_that_we_have:
            x_ = wps_dic[this_actor][0].transform.location.x
            y_ = wps_dic[this_actor][0].transform.location.y
            # map_z = get_z(world,x_,y_)
            wps_dic[this_actor][0].transform.location.z = get_z(world,x_,y_) + 0.5


        for this_actor_id in all_ids_that_we_have:
            next_waypoint = wps_dic[this_actor_id][1]    # 如果不这么写，就会无法巡线，也会出现index超出错误，至于为什么，参考one_car_test.py的实现

        total_frame = 100
        for this_frame in range(total_frame):
            start_time = time.time()
            print(f'目前的场景帧：{this_frame}')

            '''
            所有车
            '''
            for this_car in all_ids_that_we_have:
                '''
                这个车要生成
                '''
                if this_car in actor_spawn_time and actor_spawn_time[this_car] == this_frame:
                    cybertruck_bp.set_attribute('role_name',str(this_car)) # 将这个actor与id相挂钩，方便之后直接操控相应id的actor
                    this_car_actor = world.spawn_actor(cybertruck_bp, wps_dic[this_car][0].transform)
                    # name = this_car_actor.attributes['role_name']
                    # print(f'生成的这个车的role_name:{name}')           # 这里测试了 role_name 成功设置上了
                    PID=VehiclePIDController(this_car_actor,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
                    pid_dict[this_car] = PID
                    # # next = wps_dic[every_car][1] # 这一行是否有用
                    chapter_actor_list.append(this_car_actor)
                    

                '''
                这个车要移动
                '''
               


                all_actor_this_moment = world.get_actors()
                for this_actor_this_moment in all_actor_this_moment:
                    
                    # if this_actor_this_moment.attributes['role_name'] == str(this_car):
                    if 'role_name' in this_actor_this_moment.attributes and this_actor_this_moment.attributes['role_name'] == str(this_car): 
                        '''
                        role_name' in this_actor_this_moment.attributes 
                        终于明白为什么要加这一行了，因为
                        '''
                        ########################################################################################
                        this_actor_this_moment_id = int(this_actor_this_moment.attributes['role_name'])
                        PID = pid_dict[this_actor_this_moment_id]

                        next_waypoint = wps_dic[this_actor_this_moment_id][this_frame - 0]
                        this_car_transform = this_actor_this_moment.get_transform()
                        dist_to_next_waypoint = distance_vehicle(next_waypoint, this_car_transform)

                        control = PID.run_step(target_speed, next_waypoint)
                        
                        if dist_to_next_waypoint > 2: 
                            this_actor_this_moment.apply_control(control)
                        ########################################################################################

                        ########################################################################################
                        # this_actor_this_moment_id = int(this_actor_this_moment.attributes['role_name'])
                        # PID = pid_dict[this_actor_this_moment_id]

                        
                        # this_car_transform = this_actor_this_moment.get_transform()
                        # dist_to_next_waypoint = distance_vehicle(next_waypoint, this_car_transform)

                        # control = PID.run_step(target_speed, next_waypoint)
                        # this_actor_this_moment.apply_control(control)

                        # if dist_to_next_waypoint < 1.5: 
                        #     next_waypoint = wps_dic[this_actor_this_moment_id][this_frame - 0]
                        #     control = PID.run_step(target_speed, next_waypoint)
                        ########################################################################################


                



                '''
                这个车要消失
                '''
                # all_actor_this_moment = world.get_actors() # 可以删除
                # for this_actor_this_moment in all_actor_this_moment:
                #     if actor_destroy_time[this_car] == this_frame:
                #         this_actor_this_moment.destroy()

            world.tick()

            # # 计算帧的执行时间
            elapsed_time = time.time() - start_time
            # 等待剩余时间以模拟真实时间推进
            time.sleep(max(0, settings.fixed_delta_seconds - elapsed_time))
                




        

            # world.wait_for_tick()













    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print('同步模式关闭')
        for every_actor in chapter_actor_list:
            every_actor.destroy()











if __name__ =='__main__':
    try:
        main()

    except Exception as e:
        print(f'main 函数出现错误：{e}')