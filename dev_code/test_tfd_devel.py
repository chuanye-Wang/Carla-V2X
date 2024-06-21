#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time
import pdb
import numpy as np
import math
import pandas as pd
import pymap3d as pm

try:
    sys.path.append(glob.glob('../carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg'))
except IndexError:
    pass

sys.path.append("/home/ubuntu/WCY/carla-dev/PythonAPI/carla/")

import carla
from carla import Waypoint
from carla import VehicleLightState as vls
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector

import argparse
import logging
from numpy import random

import pyproj

def transform(matrix_val, lat, lon):
    """
    Transform from carla.GeoLocation to carla.Location (left_handed z-up).

    Numerical error may exist. Experiments show error is about under 1 cm in Town03.
    """
    geoloc = np.array(
        [lat, lon, 0, 1])
    loc = matrix_val.dot(geoloc.T)
    carla_loc = carla.Location(loc[0], loc[1], loc[2])
    return carla_loc.x, carla_loc.y


"""
 作者： 钟佳儒
 注释作者： 王川页
 日期： 2024.6.17
 邮箱： 553503540@qq.com
 电话： 13601193495
"""

def cs_transform(x1, y1):

    ref_x, ref_y = -40251.76572214719, 326531.9706723457  # 地理位置做的偏移量

    x_tmp, y_tmp = x1 - ref_x, y1 - ref_y # 还原真实 UTM 坐标，之后还需转换成经纬度坐标
    # print(f"数据集的UTM: {x_tmp} {y_tmp}")
    # p1 = pyproj.Proj("+proj=utm +lat_0=0 +lon_0=117 +k=1 +x_0=500000 +y_0=0 +unit=m +type=crs", preserve_units=False) # 这一步定义了一个 UTM 投影坐标系对象，后面是一些精准度配置，无关紧要
    p1 = pyproj.Proj("+proj=tmerc +lat_0=39.788261 +lon_0=116.534302", preserve_units=False)
    """
    下面这一部分是将真实轨迹的 UTM 值转换成真实经纬度
    """
    car_real_lon, car_real_lat = p1(x_tmp, y_tmp, inverse=True) # 将上述定义的 UTM 对象转换成经纬度表示
    # car_real_lon, car_real_lat = p1(476156.41,12934593.84,inverse=True) # 将上述定义的 UTM 对象转换成经纬度表示
    # print(f"数据集真实地理位置：{car_real_lon} {car_real_lat}")

    car_real_alt = 0 
    origin_lat = 39.47177     # 这部分应该替换成xodr文件中的纬度（latitude）
    origin_lon = 116.32035		# longitude经度，同上
    '''
    我的数据
    origin_lat = 39.788261 
    origin_lon = 116.534302	
    '''
    '''
    佳儒学长的数据
    origin_lat = 39.803799
    origin_lon = 116.528549

    google地图上估摸的数据
    北纬， lat = 39.47177
    东经， lon = 116.32035

    亦庄路口9的经纬度   39.48193 116.32099
    北纬， lat = 39.48193
    东经， lon = 116.3210
    
    '''
    origin_alt = 0.0
    
    """
    路径 
    /home/ubuntu/WCY/carla-dev/Unreal/CarlaUE4/Content/CustomMaps/big_map/OpenDrive/big_map.xodr
    亦庄9号路口
    """
    
    # x2, y2, z2 = pm.geodetic2enu(car_real_lat, car_real_lon, car_real_alt, origin_lat, origin_lon, origin_alt)
    x2 = car_real_lat
    y2 = car_real_lon
    print(f"carla坐标系的 x= {x2/1000}km  y= {-y2/1000}km")
    
    return x2, -y2  #carla是左手坐标系，而我们常用的是右手坐标系，因此y2要取负

def df_interpolate(df):

    ids = df.id.unique()
    interp_df = pd.DataFrame(columns=df.columns)
    for id in ids:
        df_id_veh = df.loc[df.id == id]
        t_min, t_max = min(df_id_veh.timestamp), max(df_id_veh.timestamp)
        if int((t_max - t_min) * 10) > len(df_id_veh):
            df_full = pd.DataFrame(np.linspace(t_min, t_max, num=int((t_max - t_min) * 10)+1))
            df_full.columns = ['timestamp']
            df_full.timestamp = df_full.timestamp.map(lambda x: np.round(x, 1))
            df_id_veh = pd.merge(df_id_veh, df_full, on='timestamp', how='right')
            df_id_veh.interpolate(method='linear', limit_area='inside', inplace=True)
        interp_df = interp_df.append(df_id_veh, ignore_index=True)

    return interp_df

def get_actor_blueprints(world, filter, generation):
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
    
    tfd_dir = '/home/ubuntu/WCY/carla-dev/tfd/test9/10038.csv'  # 找到数据集
    raw_df = pd.read_csv(tfd_dir) # 用 panda 读取数据集，pd的数据结构
    df = df_interpolate(raw_df)
    scene_t_min, scene_t_max = min(df.timestamp), max(df.timestamp)

    veh_ids = df.loc[df.type == 'VEHICLE'].id.unique()
    bic_ids = df.loc[df.type == 'BICYCLE'].id.unique()
    ped_ids = df.loc[df.type == 'PEDESTRIAN'].id.unique()

    ids = df.id.unique()
    actor_spawn = {}
    actor_destroy = {}
    for i in ids:
        df_id = df.loc[df.id == i]
        t_min, t_max = min(df_id.timestamp), max(df_id.timestamp)
        actor_spawn[i], actor_destroy[i] = int(10 * (t_min - scene_t_min)), int(10 * (t_max - scene_t_min))

    # PID
    args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
    args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}
    target_speed = 60
    #PID=VehiclePIDController(ego,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
    
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        # default='0.0.0.0',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=100,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-nv', '--number-of-vehicles',
        metavar='NV',
        #default=len(veh_ids),
        default=1,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-nb', '--number-of-bicycles',
        metavar='NB',
        #default=len(bic_ids),
        default=0,
        type=int,
        help='Number of bicycles (default: 10)')
    argparser.add_argument(
        '-np', '--number-of-pedestrians',
        metavar='NP',
        #default=len(ped_ids),
        default=0,
        type=int,
        help='Number of pedestrians (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterb',
        metavar='PATTERN',
        default='vehicle.harley-davidson.low_rider',
        help='Filter bycycle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationb',
        metavar='G',
        default='All',
        help='restrict to certain bicycle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000, #原本有的
        # default=2000, #自己加的
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable car lights')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    args.number_of_vehicles = len(veh_ids) + 1

    vehicles_list = []
    pedestrians_list = []
    all_id = []
    print("开始连接carla服务器")
    client = carla.Client('127.0.0.1',2000)
    client.set_timeout(2.0)
    print("成功连接")
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()

        debug = world.debug
        debug.draw_point(carla.Location(135.7, -81.9, 0.0), life_time=0)
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        synchronous_master = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps  = 10
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        blueprintsVeh = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsBic = get_actor_blueprints(world, args.filterb, args.generationb)
        blueprintsPed = get_actor_blueprints(world, args.filterw, args.generationw)

        blueprintsVeh = [x for x in blueprintsVeh if int(x.get_attribute('number_of_wheels')) == 4]
        blueprintsVeh = [x for x in blueprintsVeh if x.has_attribute('role_name')]
        blueprintsBic = [x for x in blueprintsBic if x.has_attribute('role_name')]
        blueprintsVeh = sorted(blueprintsVeh, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor
        #pdb.set_trace()
        # --------------
        # Spawn vehicles
        # --------------
        hero = args.hero
        #for n, transform in enumerate(spawn_points):
        '''
        tfd_dir = '/home/ubuntu/WCY/carla-dev/tfd/test9/10038.csv'  # 找到数据集
        raw_df = pd.read_csv(tfd_dir) # 用 panda 读取数据集，pd的数据结构
        df = df_interpolate(raw_df)
        scene_t_min, scene_t_max = min(df.timestamp), max(df.timestamp)

        veh_ids = df.loc[df.type == 'VEHICLE'].id.unique()
        bic_ids = df.loc[df.type == 'BICYCLE'].id.unique()
        ped_ids = df.loc[df.type == 'PEDESTRIAN'].id.unique()
        
        '''
        '''
        下面这段代码根本没用上，因为for循环进去就break了，因此为了方便读代码直接注释掉了
        
        '''
        # for n in range(len(veh_ids)):  # 对于每一个数据集中标记为vehicle的id个数，也就是汽车的数量
        #     break
        #     if n >= args.number_of_vehicles:
        #         break

        #     blueprint = random.choice(blueprintsVeh)
        #     if blueprint.has_attribute('color'):
        #         color = random.choice(blueprint.get_attribute('color').recommended_values)
        #         blueprint.set_attribute('color', color)
        #     if blueprint.has_attribute('driver_id'):
        #         driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
        #         blueprint.set_attribute('driver_id', driver_id)
        #     if hero:
        #         blueprint.set_attribute('role_name', 'hero')
        #         hero = False
        #     else:
        #         blueprint.set_attribute('role_name', 'autopilot')

        #     # prepare the light state of the cars to spawn
        #     light_state = vls.NONE
        #     if args.car_lights_on:
        #         light_state = vls.Position | vls.LowBeam | vls.LowBeam

        #     # spawn the cars and set their autopilot and light state all together

        #     x1, y1 = df.loc[df.id == veh_ids[n]].x.values[0], df.loc[df.id == veh_ids[n]].y.values[0]
        #     #x1, y1 = 416766, 4732399
        #     x2, y2 = cs_transform(x1, y1)
	    
        #     # spawn the cars and set their autopilot and light state all together
        #     location = carla.Location(x=x2, y=y2, z=2.0)
        #     #location_ego = carla.Location(x=0.0,y=0.0,z=0.0)
        #     rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        #     transform = carla.Transform(location, rotation)

        #     batch.append(SpawnActor(blueprint, transform)
        #         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        # -------------
        # Spawn pedestrians
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 1     # how many pedestrians will walk through the road
        wps_dict = {}

        '''

        ids = df.id.unique()
        actor_spawn = {}
        actor_destroy = {}
        for i in ids:
            df_id = df.loc[df.id == i]
            t_min, t_max = min(df_id.timestamp), max(df_id.timestamp)
            actor_spawn[i], actor_destroy[i] = int(10 * (t_min - scene_t_min)), int(10 * (t_max - scene_t_min))

        '''
        for id in ids: # 所有id，包括vehicle、bicycle、pedestrains的id

            df_id = df.loc[df.id == id] # 把同一个id的所有数据提取到同一个dataframe中
            
            wps = []
            for t in range(len(df_id)): # 循环一个id的所有数据点数量的次数
                x_, y_ = cs_transform(df_id.x.values[t], df_id.y.values[t])
                location = carla.Location(x=x_, y=y_, z=0.5)
                rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
                transform = carla.Transform(location, rotation)

                wps.append(world.get_map().get_waypoint(transform.location)) # 这个get_waypoint不应该调用，
                # 因为这个函数是根据我数据点的位置，找离它最近的道路中心线上的way_point,这样做就有悖于我们想要
                # 做真实车辆数据集再现的目的了
            wps_dict[id] = wps

        pid_dict = {}
        controller_dict = {}

        for t in range(args.frames): # args.frames = 100    100帧因为10秒，10Hz
            print(f"t=_{t}")
            actor_list = world.get_actors()
            
            for id in veh_ids:  
                
                if id in actor_spawn and actor_spawn[id] == t:
                    x_t, y_t = cs_transform(df.loc[df.id == id].x.values[0], df.loc[df.id == id].y.values[0])
                    location = carla.Location(x=x_t, y=y_t, z=0.01)  # 生成车辆x，y，z
                    theta_t = math.degrees(df.loc[df.id == id].theta.values[0])
                    #theta_t = df.loc[df.id == id].theta.values[0]
                    rotation = carla.Rotation(pitch=0.0, yaw=theta_t, roll=0.0)
                    transform = carla.Transform(location, rotation)

                    blueprint = random.choice(blueprintsVeh)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                    blueprint.set_attribute('role_name', str(id))
                    
                    try:
                        actor = world.spawn_actor(blueprint, transform)
                        PID=VehiclePIDController(actor,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
                        pid_dict[id] = PID
                    except:
                        #raise Exception
                        continue
                    print(f"generate_veh_{id}")
                
                actor_list = world.get_actors()
                if id in actor_destroy and actor_destroy[id] == t:
                    
                    for actor in actor_list:
                        
                        if 'role_name' in actor.attributes and actor.attributes['role_name'] == str(id):
                            
                            carla.command.DestroyActor(actor) # 先屏蔽掉看看效果

                            print(f"destroy_veh_{id}")
                            break
                
                if id in actor_spawn and actor_destroy[id] <= t:
                    continue
                actor_list = world.get_actors()
                for actor in actor_list:
                    if 'role_name' in actor.attributes and actor.attributes['role_name'] == str(id):
                       
                        try:
                            
                            PID = pid_dict[id]
                            next_wp = wps_dict[id][t - actor_spawn[id]]
                            actor_dist = distance_vehicle(next_wp, actor.get_transform())
                            control = PID.run_step(target_speed, next_wp)

                            if actor_dist > 10:
                                actor.apply_control(control)
                                print(f'move_veh_{id}')
                            break

                        except:
                            print(f'move_error_veh_{id}')
                            continue
            
            for id in bic_ids:
                if id in actor_spawn and actor_spawn[id] == t:
                    x_t, y_t = cs_transform(df.loc[df.id == id].x.values[0], df.loc[df.id == id].y.values[0])
                    location = carla.Location(x=x_t, y=y_t, z=0.5)
                    theta_t = math.degrees(df.loc[df.id == id].theta.values[0])
                    rotation = carla.Rotation(pitch=0.0, yaw=theta_t, roll=0.0)
                    transform = carla.Transform(location, rotation)

                    blueprint = random.choice(blueprintsBic)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                    blueprint.set_attribute('role_name', str(id))
                    try:
                        actor = world.spawn_actor(blueprint, transform)
                        PID=VehiclePIDController(actor,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
                        pid_dict[id] = PID
                    except:
                        #raise Exception
                        continue
                    print(f"generate_bic_{id}")

                actor_list = world.get_actors()
                if id in actor_destroy and actor_destroy[id] == t:
                    for actor in actor_list:
                        if 'role_name' in actor.attributes and actor.attributes['role_name'] == str(id):
                            carla.command.DestroyActor(actor) # 先删掉看看效果
                            print(f"destroy_bic_{id} 修改了")
                            break
                
                if id in actor_spawn and actor_destroy[id] <= t:
                    continue
                actor_list = world.get_actors()
                for actor in actor_list:
                    if 'role_name' in actor.attributes and actor.attributes['role_name'] == str(id):
    
                        try:
                            PID = pid_dict[id]
                            next_wp = wps_dict[id][t - actor_spawn[id]]
                            actor_dist = distance_vehicle(next_wp, actor.get_transform())
                            control = PID.run_step(target_speed, next_wp)
                            #actor.apply_control(control)
                            if actor_dist > 3:
                                actor.apply_control(control)
                                print(f'move_bic_{id}')
                            break

                        except:
                            print(f'move_error_bic_{id}')
                            continue
            
            world.tick()
            if t == 99:
                
                for k in ids:
                    for actor in actor_list:
                        if 'role_name' in actor.attributes and actor.attributes['role_name'] == str(k):
                            carla.command.DestroyActor(actor) # 先注释掉看看
                            print(f"final_destroy_{k},修改过")
                            break

    finally:
        
        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d pedestrians' % len(pedestrians_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
