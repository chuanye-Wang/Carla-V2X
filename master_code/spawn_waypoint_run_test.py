import numpy as np
import pandas as pd
import sys,os
import pyproj

sys.path.append("C:/Users/55350/Desktop/WindowsNoEditor/PythonAPI/carla")

import carla
import math
import random
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector, is_within_distance

# from agents.navigation.controller import VehiclePIDController


def main():
    try:
        print("初始化Client端口")
        client = carla.Client('127.0.0.1',2000)
        client.set_timeout(5.0)
        print("初始化完成")

    except carla.TimeoutError as T_o:
        print(f'初始化超时：{T_o}')

    try:
        world = client.get_world()
        client.set_timeout(5.0)
    except Exception as e:
        print(f'获取世界报错：{e}')

    map = world.get_map()
    start_point = map.get_spawn_points()[98].location
    destination = map.get_spawn_points()[27].location

    world.debug.draw_point(start_point, size=0.2, color=carla.Color(r=0, g=0, b=255), life_time=60.0)
    world.debug.draw_point(destination, size=0.2, color=carla.Color(r=0, g=0, b=255), life_time=60.0)
    # print(f'出发点：{start_point}')
    # print(f'终点：{destination}')

    distance = 2
    grp = GlobalRoutePlanner(map, distance)
    route = grp.trace_route(start_point, destination)

    # try:
    #     wps = []
    #     for waypoint, car_operation in route:
    #         loc = waypoint.transform.location
    #         world.debug.draw_point(loc,size=0.2,color=carla.Color(r=255,g=0,b=0),life_time=60)
        
    
    # except Exception as e:
    #     print(f'waypoint绘制错误：{e}')

    try:
        wps = []
        for i in range(len(route)):
            wps.append(route[i][0])
        draw_waypoints(world, wps)

        
        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        ego = world.spawn_actor(ego_bp, map.get_spawn_points()[98])

        args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
        args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}
            
        PID=VehiclePIDController(ego,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)

        target_speed = 60
        next = wps[0]
        i=0
        while True:
            if i == len(wps)-1:
                break
            ego_transform = ego.get_transform()
            ego_dist = distance_vehicle(next, ego_transform)
            control = PID.run_step(target_speed, next)
            ego.apply_control(control)
            if ego_dist < 1.9: 
                i = i + 1
                next = wps[i]
                control = PID.run_step(target_speed, next)
            
            

            
            world.wait_for_tick()
    
    finally:
        ego.destroy()



    


if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print(f'main函数报错：{e}')

