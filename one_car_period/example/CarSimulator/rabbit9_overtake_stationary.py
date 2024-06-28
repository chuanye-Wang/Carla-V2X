# step9: update route when front stationary vehicle 
# 9.1 overtake stationary vehicle 
# 9.2 ACC 
import os
import sys

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
import numpy as np
import pygame
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints, distance_vehicle, vector, is_within_distance, get_speed

distance = 2.0

client = carla.Client('localhost',2000)
world = client.get_world()
m = world.get_map()
transform = carla.Transform()
spectator = world.get_spectator()
bv_transform = carla.Transform(transform.location + carla.Location(z=200,x=0), carla.Rotation(yaw=0, pitch=-90))
spectator.set_transform(bv_transform)

blueprint_library = world.get_blueprint_library()
spawn_points = m.get_spawn_points()

T = 10
for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=T)
    world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=T)

# global path planner 
origin = carla.Location(spawn_points[117].location)
destination = carla.Location(spawn_points[51].location)        

grp = GlobalRoutePlanner(m, distance)
route = grp.trace_route(origin, destination)

wps = []
for i in range(len(route)):
    wps.append(route[i][0])
draw_waypoints(world, wps)

for pi, pj in zip(route[:-1], route[1:]):
    pi_location = pi[0].transform.location
    pj_location = pj[0].transform.location 
    pi_location.z = 0.5
    pj_location.z = 0.5
    world.debug.draw_line(pi_location, pj_location, thickness=0.2, life_time=T, color=carla.Color(b=255))
    pi_location.z = 0.6
    world.debug.draw_point(pi_location, color=carla.Color(b=255), life_time=T)   
    
# spawn ego vehicle
ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
ego = world.spawn_actor(ego_bp, spawn_points[117])

# spawn stationary target vehicle 
target_bp = blueprint_library.find('vehicle.tesla.model3')
target = world.spawn_actor(target_bp, spawn_points[49])


# https://carla.readthedocs.io/en/docs-preview/python_api/#carla.Map.get_waypoint
# This recipe shows the current traffic rules affecting the vehicle. 
# Shows the current lane type and if a lane change can be done in the actual lane or the surrounding ones.

waypoint = world.get_map().get_waypoint(ego.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
print("Current lane type: " + str(waypoint.lane_type))
# Check current lane change allowed
print("Current Lane change:  " + str(waypoint.lane_change))
# Left and Right lane markings
print("L lane marking type: " + str(waypoint.left_lane_marking.type))
print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
print("R lane marking type: " + str(waypoint.right_lane_marking.type))
print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))


# PID
#args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}
args_lateral_dict = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}

#args_long_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05}
args_long_dict = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}

PID=VehiclePIDController(ego,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)
#obj_PID=VehiclePIDController(target,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)

i = 0
target_speed = 30
next = wps[0]

#j = 0
#obj_speed = 10
#obj_next = obj_wps[0]

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))

# camera 
camera_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_trans, attach_to=ego)

camera.listen(lambda image: pygame_callback(image, renderObject))

# Get camera dimensions
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Instantiate objects for rendering and vehicle control
renderObject = RenderObject(image_w, image_h)

# Initialise the display
pygame.init()
gameDisplay = pygame.display.set_mode((image_w,image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
# Draw black to the display
gameDisplay.fill((0,0,0))
gameDisplay.blit(renderObject.surface, (0,0))
pygame.display.flip()


try:
    while True:
        ego_transform = ego.get_transform()
        spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=80), carla.Rotation(pitch=-90)))
        ego_loc = ego.get_location()
        world.debug.draw_point(ego_loc, color=carla.Color(r=255), life_time=T)
        world.debug.draw_point(next.transform.location, color=carla.Color(r=255), life_time=T)      
        ego_dist = distance_vehicle(next, ego_transform)
        #ego_vect = vector(ego_loc, next.transform.location)
        control = PID.run_step(target_speed, next)
        
        target_loc = target.get_location()
        obj_transform = target.get_transform()
        #spectator.set_transform(carla.Transform(obj_transform.location + carla.Location(z=80), carla.Rotation(pitch=-90)))    
        #obj_control = obj_PID.run_step(obj_speed, obj_next)
        #obj_dist = distance_vehicle(obj_next, obj_transform)
    
        if i == (len(wps)-1):
            control = PID.run_step(0, wps[-1])
            ego.apply_control(control)
            print('this trip finish')
            break
    
        if ego_dist < 1.5: 
            i = i + 1
            next = wps[i]
            control = PID.run_step(target_speed, next)
        
        # lane change start waypoint    
        if is_within_distance(target.get_transform(), ego.get_transform(), 20, [-50, 50]):
            ego_loc = ego.get_location()
            current_w = world.get_map().get_waypoint(ego_loc)
            d_lanechange = 1*target_speed/3.6
            next_w = current_w.next(d_lanechange)[0]
            
            '''
            while distance_vehicle(next_w.get_right_lane(), ego_transform) < 0.5:
                ego_transform = ego.get_transform()
                control = PID.run_step(target_speed, next_w.get_right_lane())
                ego.apply_control(control)
            '''
            
            if next_w.lane_change == carla.LaneChange.Right or next_w.lane_change == carla.LaneChange.Both:
                origin = next_w.get_right_lane().transform.location
                #destination = next_w.next(20)[0].get_right_lane().transform.location
                destination = carla.Location(spawn_points[52].location)        

                grp = GlobalRoutePlanner(m, distance)
                route = grp.trace_route(origin, destination)
            
                wps = []
                for j in range(len(route)):
                    wps.append(route[j][0])
                
                for pi, pj in zip(route[:-1], route[1:]):
                    pi_location = pi[0].transform.location
                    pj_location = pj[0].transform.location 
                    pi_location.z = 0.5
                    pj_location.z = 0.5
                    world.debug.draw_line(pi_location, pj_location, thickness=0.2, life_time=T, color=carla.Color(b=255))
                    pi_location.z = 0.6
                    world.debug.draw_point(pi_location, color=carla.Color(b=255), life_time=T)   
            
                i = 1
                next = wps[i]                
            
        '''
        if is_within_distance(target.get_transform(), ego.get_transform(), 40, [-25, 25]):    
           speed = min(target_speed, obj_speed, ego.get_speed_limit())
           control = PID.run_step(speed, next)
        '''
        if is_within_distance(target.get_transform(), ego.get_transform(), 15, [-25, 25]):
            control.throttle = 0.0
            control.brake = 0.5
            control.hand_brake = False
        '''
        if j == (len(obj_wps)-1):
            obj_control = obj_PID.run_step(0, obj_wps[-1])
            target.apply_control(obj_control)
            print('target the trip finish')
            break
    
        if obj_dist < 1.5: 
            j = j + 1
            obj_next = obj_wps[j]
            obj_control = obj_PID.run_step(obj_speed, obj_next)
        
        target.apply_control(obj_control)
        '''
        
        ego.apply_control(control)
        world.wait_for_tick()
        
        # Update the display
        gameDisplay.blit(renderObject.surface, (0,0))
        pygame.display.flip()
        
        
finally:
    ego.destroy()
    target.destroy()
    camera.stop()
    pygame.quit()     
