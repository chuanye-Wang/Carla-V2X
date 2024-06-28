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
from agents.navigation.controller import VehiclePIDController



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
    car_real_lon, car_real_lat, car_real_alt= p1(x_tmp, y_tmp, inverse=True), 0

    # 这部分应该用carla世界中原点的真实经纬度代替
    origin_lon = 116.729
    origin_lat = 39.8664
    origin_alt = 0
    # 转换后的坐标
    x_rst, y_rst, z_rst = pm.geodetic2enu(car_real_lat, car_real_lon, car_real_alt, origin_lat, origin_lon, origin_alt)
    y_rst = -y_rst
    print(f'{x_rst}  {y_rst}')

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


def get_actor_blueprints(world, filter, generation):
    """
    @brief 获取符合特定过滤条件和生成代的actor蓝图
    
    @param world CARLA仿真世界对象
    @param filter 用于过滤蓝图的字符串
    @param generation 生成代

    @return 符合过滤条件和生成代的actor蓝图列表
    """

    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []



def main():
    path2dataset_file = "10039.csv"
    # print(f'dataset path: {path2dataset_file}')
    raw_data = pd.read_csv(path2dataset_file) # 只不过是换成pd的数据结构读取了而已
    colums_we_want = ['timestamp','id','type','x','y', 'theta'] # 只选择这些节省时间和空间
    raw_data = raw_data[colums_we_want] # 这才是我们确切需要的，不带任何冗余的数据
    # print(raw_data)

    processed_data = df_interpolate(raw_data) # 数据预处理，补齐缺失帧

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
    
    # 随便选了一个id试验了一下，可以的
    # print(actor_spawn_time[2473222]) # 9.5
    # print(actor_destroy_time[2473222]) # 9.900000095367432


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
        settings.max_substep_delta_time = 0.01 # 设置最大子步的步长
        settings.max_substeps  = 10 # 最大子步数，将每一时间步长分解为最多10个子步，具体分解为多少个子步由UE4计算情况动态决定
        
        # settings.synchronous_mode = True 启用同步模式后，所有客户端都需要遵循同步规则
        # synchronous_master = True 确定了此客户端脚本负责发送同步步长推进命令，驱动整个仿真环境前进
        synchronous_master = True
        settings.synchronous_mode = True 

        # 因为数据集是10Hz的
        settings.fixed_delta_seconds = 0.1
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
         # 将插值处理过后的原始数据集中的所有 “此循环时的id” 的数据打包出来成为一个单独的（暂时的）数据结构
        all_data_of_this_id = processed_data.loc[processed_data.id == this_specific_id]

        wps = [] # 路径点数列

        # if(actor_spawn_time[this_specific_id] == this_frame):
        # 把这个id的数据集转换之后存在 wps 数列当中
        for this_frame in range(len(all_data_of_this_id)): 
            x_, y_ = cs_transform(all_data_of_this_id.x.values[this_frame], all_data_of_this_id.y.values[this_frame])

            location = carla.Location(x=x_, y=y_, z=0.1)
            rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            transform = carla.Transform(location, rotation)

            wps.append(world.get_map().get_waypoint(transform.location))

        wps_dic[this_specific_id] = wps


    '''
    上一步已经获得了所有这个id对应的轨迹点，接下来debug打印出来看看对不对
    '''


    print(wps_dic)

    # blueprintsVeh = get_actor_blueprints(world, 'vehicle.*', 'All')
    # blueprintsBic = get_actor_blueprints(world, 'vehicle.harley-davidson.low_rider', 'All')
    # blueprintsPed = get_actor_blueprints(world, 'walker.pedestrian.*', 'All')

    # blueprintsVeh = [x for x in blueprintsVeh if int(x.get_attribute('number_of_wheels')) == 4]
    # blueprintsVeh = [x for x in blueprintsVeh if x.has_attribute('role_name')]
    # blueprintsBic = [x for x in blueprintsBic if x.has_attribute('role_name')]
    # blueprintsVeh = sorted(blueprintsVeh, key=lambda bp: bp.id) # 对蓝图id进行排序，实验的时候控制变量作用，前后对比方便
    # '''
    # 蓝图处理
    # 接下来是生成actor的函数
    # world_tik
    # 然后是销毁的actor的函数
    # '''

    # '''
    # 这一部分的基本想法是：【每一帧】：
    #                                 object要生成
    #                                 object要销毁
    #                                 object要移动
    # '''
    # all_veh_ids = processed_data.loc[processed_data.type == 'VEHICLE'].id.unique()
    # all_bic_ids = processed_data.loc[processed_data.type == 'BICYCLE'].id.unique()
    # all_ped_ids = processed_data.loc[processed_data.type == 'PEDESTRIAN'].id.unique()

    # # PID
    # args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
    # args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}
    # target_speed = 60

    # pid_dict = {}
    # controller_dict = {}

    # totol_frame = 100
    # for this_frame in range(totol_frame):
    #     # actor_spawn_time = {}   # 保存了每个ID在仿真开始多久之后 生成 的时间
    #     # actor_destroy_time = {} # 保存了每个ID在仿真开始多久之后 消失 的时间
    #     print(f'仿真时间戳：{this_frame}')
    #     for this_vehicle in all_veh_ids:
    #         if actor_spawn_time[this_vehicle] == this_frame:
    #             # 计算出初始位置，之所以不用前面的是因为，之前的算出来的被修改成道路中心点了，这样初始场景就不真实了
    #             x_t, y_t = cs_transform(processed_data.loc[processed_data.id == id].x.values[0], 
    #                                     processed_data.loc[processed_data.id == id].y.values[0])
    #             # processed_data = df_interpolate(raw_data) # 数据预处理，补齐缺失帧
    #             location = carla.Location(x=x_t, y=y_t, z=0.01)  # 生成车辆x，y，z
    #             theta_t = math.degrees(processed_data.loc[processed_data.id == id].theta.values[0])                
    #             rotation = carla.Rotation(pitch=0.0, yaw=theta_t, roll=0.0)
    #             transform = carla.Transform(location, rotation)

    #             blueprint = random.choice(blueprintsVeh)

    #             try:
    #                 actor = world.spawn_actor(blueprint, transform)
    #                 print(f"generate_veh_{id}")
    #                 PID = VehiclePIDController(actor,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
    #                 pid_dict[id] = PID

    #             except:
    #                 #raise Exception
    #                 pass
                    




    

    

    
                
                


























if __name__ == '__main__':
    try:
        main()
    
    except Exception as e:
        print(f'main failed: {e}')
    