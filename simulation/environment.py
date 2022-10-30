import math
import time
import random
import numpy as np
import logging
import pygame
from simulation.connection import carla, logging
from parameters import CONTINUOUS_ACTION_SPACE, VISUAL_DISPLAY
from simulation.settings import CAR_NAME, EPISODE_LENGTH, NUMBER_OF_VEHICLES, NUMBER_OF_PEDESTRIAN
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor, LaneInvasionSensor

random.seed(0)

class CarlaEnvironment():

    def __init__(self, client, world) -> None:

        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.vehicle = None
        self.settings = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None
        #self.destination = None
        self.waypoint = None
        self.trajectory = None
        self.velocity = None
        self.location = None
        self.prev_steer = None
        self.signed_waypoint_distance = None
        self.action_space = self._get_action_space()
        self.continous_action_space = CONTINUOUS_ACTION_SPACE
        self.episode_start_time = None

        # Objects to be kept alive
        self.camera_obj = None
        if VISUAL_DISPLAY:
            self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()


        self._create_pedestrians()
        logging.info("CarlaEnvironment obj has been initialized!")


    # A reset function for reseting our environment.

    def _reset(self):

        try:

            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()

            # Spawn points of the entire map!
            spawn_points = self.map.get_spawn_points()

            # Blueprint of our main vehicle
            vehicle_bp = self._get_vehicle(CAR_NAME)

            #vehicle = self.actor_vehicle(vehicle_bp, spawn_points)
            self._set_vehicle(vehicle_bp, spawn_points)
            self.actor_list.append(self.vehicle)


            # Destination set
            self._set_destination(self.vehicle.get_location(), spawn_points)

            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.001)
            self.front_camera = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            #self.third_view_obj = CameraSensorEnv(self.vehicle)
            if VISUAL_DISPLAY:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Quick start our vehicle from its initial state
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0))

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)


            # Lane Invasion sensor
            self.lane_invasion_obj = LaneInvasionSensor(self.vehicle)
            self.wrong_maneuver = self.lane_invasion_obj.wrong_maneuver
            self.sensor_list.append(self.lane_invasion_obj.sensor)

            # Quick start our vehicle from its initial state
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5))

            # Velocity
            velocity = self.vehicle.get_velocity()
            self.velocity = math.sqrt( velocity.x**2 + velocity.y**2 + velocity.z**2) / 10

            # Rotation
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()
            self.prev_steer = 0.000
            
            # Waypoint nearby angle and distance from it
            angles = list()
            distance = None
            car_angle = None
            road_angle = None
            #fw_vec = None

            self.waypoint = self.map.get_waypoint(self.location, project_to_road=True, lane_type=(carla.LaneType.Driving))
            distance = (math.sqrt((self.waypoint.transform.location.x - self.location.x)**2 + (self.waypoint.transform.location.y - self.location.y)**2))
            for i in range(5):

                self.waypoint = self.waypoint.next(1.0)[-1]
                #fw_vec = self.waypoint.transform.rotation.get_forward_vector()
                #distances.append(math.sqrt((self.waypoint.transform.location.x - self.location.x)**2 + (self.waypoint.transform.location.y - self.location.y)**2))
                if self.rotation > 90.0000:
                    car_angle = 180.0000 - self.rotation
                elif self.rotation < -90.0000:
                    car_angle = -180.0000 - self.rotation
                else:
                    car_angle = self.rotation

                if self.waypoint.transform.rotation.yaw > 90.0000:
                    road_angle = 180.0000 - self.waypoint.transform.rotation.yaw
                elif self.waypoint.transform.rotation.yaw < -90.0000:
                    road_angle = -180.0000 - self.waypoint.transform.rotation.yaw
                else:
                    road_angle = self.waypoint.transform.rotation.yaw
                if (road_angle > 0.00000 and car_angle < 0.00000):
                    angles.append((180.0000-car_angle)-(180.0000-road_angle))
                elif(road_angle < 0.00000 and car_angle > 0.00000):
                    angles.append((180.0000-car_angle)-(180.0000-road_angle))
                else:
                    angles.append((180.0000+road_angle)-(180.0000+car_angle))
            
                        
                        
                        
            #print(fw_vec)
            self.trajectory = np.median(angles) / 45
            #if fw_vec.x == -1.000000 and fw_vec.y < 0.000000:
                #self.trajectory *= -1
            #print(round(self.trajectory, 8))
            self.signed_waypoint_distance = distance #np.mean(distances)         

            self.collision_history.clear()
            

            # Time noted for the start of the episode
            self.episode_start_time = time.time()

            self.nav_data = np.array([self.trajectory, self.velocity, self.prev_steer, self.signed_waypoint_distance])

            logging.info("Environment has been resetted.")

            return self.front_camera, self.nav_data

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            if VISUAL_DISPLAY:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.


    def _step(self, action_idx):
        try:

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                self.vehicle.apply_control(carla.VehicleControl(steer=float(action_idx[0]), throttle=float(action_idx[1])))
                self.prev_steer = float(action_idx[0])
            else:
                action = self.action_space[action_idx]
                self.vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1]))
                self.prev_steer = action[1]
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data
            self.wrong_maneuver = self.lane_invasion_obj.wrong_maneuver

            # Velocity of the vehicle
            self.velocity = self.velocity/10
            
            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            new_location = self.vehicle.get_location()

            # Location of the vehicle
            #vehicle_x, vehicle_y = new_location.x, new_location.y
            #distance_covered = math.sqrt((new_location.x - self.location.x)**2 + (new_location.y - self.location.y)**2)
            self.location = new_location

            # Randomly picked next waypoint in 1.0m distance.
            # Waypoint broken down in its three necessary components.
            # Waypoint nearby angle and distance from it
            angles = list()
            distances = list()
            car_angle = None
            road_angle = None
            #fw_vec = None

            #self.waypoint = self.map.get_waypoint(self.location, project_to_road=True, lane_type=(carla.LaneType.Driving))
            #distance = (math.sqrt((self.waypoint.transform.location.x - self.location.x)**2 + (self.waypoint.transform.location.y - self.location.y)**2))
            for i in range(5):

                self.waypoint = self.waypoint.next(1.0)[-1]
                #fw_vec = self.waypoint.transform.rotation.get_forward_vector()
                distances.append(math.sqrt((self.waypoint.transform.location.x - self.location.x)**2 + (self.waypoint.transform.location.y - self.location.y)**2))
                if self.rotation > 90.0000:
                    car_angle = 180.0000 - self.rotation
                elif self.rotation < -90.0000:
                    car_angle = -180.0000 - self.rotation
                else:
                    car_angle = self.rotation

                if self.waypoint.transform.rotation.yaw > 90.0000:
                    road_angle = 180.0000 - self.waypoint.transform.rotation.yaw
                elif self.waypoint.transform.rotation.yaw < -90.0000:
                    road_angle = -180.0000 - self.waypoint.transform.rotation.yaw
                else:
                    road_angle = self.waypoint.transform.rotation.yaw
                if (road_angle > 0.00000 and car_angle < 0.00000):
                    angles.append((180.0000-car_angle)-(180.0000-road_angle))
                elif(road_angle < 0.00000 and car_angle > 0.00000):
                    angles.append((180.0000-car_angle)-(180.0000-road_angle))
                else:
                    angles.append((180.0000+road_angle)-(180.0000+car_angle))
            self.trajectory = np.median(angles) / 45
            #if fw_vec.x == -1.000000 and fw_vec.y < 0.000000:
            #    self.trajectory *= -1
            #print(round(self.trajectory, 8))
            self.signed_waypoint_distance = np.mean(distances) #np.mean(distances) 
            
            # Rewards are given below!
            
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -1 * self.collision_history.pop(-1)
                logging.warning("Vehicle has collided.")
            elif self.wrong_maneuver:
                done = True
                reward = -1 * (self.velocity * 36 * (abs(self.trajectory) * 90))
                logging.warning("Vehicle has gone out of the lane.")
            elif self.episode_start_time + 5 < time.time() and self.velocity * 36 < 1.0:
                logging.warning("Vehicle has stopped moving.")
                reward = -100
                done = True
            elif self.velocity * 36 > 25.0:
                logging.warning("Vehicle is moving pretty fast.")
                reward = -1 * self.velocity * 36
                done = True
            elif self.episode_start_time + EPISODE_LENGTH < time.time():
                logging.info("Times up, episode is over.")
                reward = 2000
                done = True
            

            #if self.velocity * 36 > 25.0 or self.velocity * 36 < 15.0:
                #logging.warning("Not the best driving speed: {}".format(self.velocity*36))
                #print("trajectory: ", abs(self.trajectory)," waypoint_distance: ", abs(self.signed_waypoint_distance))
                #print(self.vehicle.get_speed())
                #done = False
                #reward += -1 #* (self.velocity * 36)
                #if self.velocity * 36 > 25.0 :
                #    done = True
            #if abs(self.signed_waypoint_distance) > 2.0:
            #    logging.warning("Too far away from center of the lane: {}".format(abs(self.signed_waypoint_distance)))
            #    reward += -1
            #if abs(self.trajectory) > 1.0:
            #    logging.warning("Too far away from center of the lane: {}".format(abs(self.signed_waypoint_distance)))
            #    reward += -1
            #else:
            #done = False
            #print(car_angle)
            #print(road_angle)
            if not done:
                if self.velocity < 20.0:
                    reward = (self.velocity * 1.8) * (self.velocity * 36) - (11.25 * abs(self.trajectory))#- (abs(self.signed_waypoint_distance))
                elif self.velocity > 20.0:
                    reward = (1/(self.velocity * 1.8)) * (self.velocity * 36) - (11.25* abs(self.trajectory))# - (abs(self.signed_waypoint_distance))
                else:
                    reward = (self.velocity * 36) - (11.25* abs(self.trajectory))# - (abs(self.signed_waypoint_distance))
            #print("trajectory: ", self.trajectory)

            #elif (self.destination.location.x <= vehicle_x + 1.0 and self.destination.location.x >= vehicle_x - 1.0) and (self.destination.location.y <= vehicle_y + 1.0 and self.destination.location.y >= vehicle_y - 1.0):
            #    logging.info("Reached the destination.")
            #    reward = 10#12 * (EPISODE_LENGTH - (time.time() - self.episode_start_time))
            #    done = True
            #print("trajectory: ", abs(self.trajectory)," waypoint_distance: ", self.signed_waypoint_distance)
            # tick
            while (self.front_camera is None):
                time.sleep(0.001)

            # navigation comprises of 4 data points
            self.nav_data = np.array([np.array(self.trajectory), np.array(self.velocity), np.array(self.prev_steer), np.array(self.signed_waypoint_distance)]).reshape(4,1)

            # Remove everything that has been spawned in the env
            if done:
                for sensor in self.sensor_list:
                    sensor.destroy()
                self.remove_sensors()
                for actor in self.actor_list:
                    actor.destroy()

            return self.front_camera, self.nav_data, reward, done, None

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            if VISUAL_DISPLAY:
                pygame.quit()



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!


    def _create_pedestrians(self):
        try:
            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # 3. Starting the motion of our pedestrians
            # set how many pedestrians can cross the road
            self.world.set_pedestrians_cross_factor(30)
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

            logging.info("NPC pedestrians / walkers have been generated.")
        except:
            logging.info(
                "Unfortunately, we couldn't create pedestrians in our world.")
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def _set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            logging.warning(
                "Unfortunately, we couldn't create other ai vehicles in our world.")
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.


    def _change_town(self, new_town):
        self.world = self.client.load_world(new_town)

    # Getter for fetching the current state of the world that simulator is in.
    def _get_world(self) -> object:
        return self.world

    # Getter for fetching blueprint library of the simulator.
    def _get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()

    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!

    def _get_action_space(self):
        action_space = \
            np.array([
                [1.0,   0.0], #Acceleration: 1.0 , Steering:  0.0
                [1.0, -0.30], #Acceleration: 1.0 , Steering: -0.3
                [1.0,  0.30], #Acceleration: 1.0 , Steering: +0.3
                [1.0, -0.10], #Acceleration: 1.0 , Steering: -0.1
                [1.0,   0.1], #Acceleration: 1.0 , Steering: +0.1 
                [0.5,   0.0], #Acceleration: 0.5 , Steering:  0.0
                [0.5, -0.30], #Acceleration: 0.5 , Steering: -0.3
                [0.5,  0.30], #Acceleration: 0.5 , Steering: +0.3
                [0.5, -0.10], #Acceleration: 0.5 , Steering: -0.1
                [0.5,   0.1], #Acceleration: 0.5 , Steering: +0.1        
                [0.1,   0.0], #Acceleration: 0.1 , Steering:  0.0
                [0.1, -0.30], #Acceleration: 0.1 , Steering: -0.3
                [0.1,  0.30], #Acceleration: 0.1 , Steering: +0.3
                [0.1, -0.10], #Acceleration: 0.1 , Steering: -0.1
                [0.1,   0.1], #Acceleration: 0.1 , Steering: +0.1
                [0.0,   0.0], #Acceleration: 0.0 , Steering:  0.0
                [0.0, -0.30], #Acceleration: 0.0 , Steering: -0.3
                [0.0,  0.30], #Acceleration: 0.0 , Steering: +0.3
                [0.0, -0.10], #Acceleration: 0.0 , Steering: -0.1
                [0.0,   0.1], #Acceleration: 0.0 , Steering: +0.1
                [0.5,   0.5], #Acceleration: 0.5 , Steering: +0.5
                [0.5,  -0.5], #Acceleration: 0.5 , Steering: -0.5
                [1.0,   0.5], #Acceleration: 1.0 , Steering: +0.5
                [1.0,  -0.5], #Acceleration: 1.0 , Steering: -0.5
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def _get_vehicle(self, vehicle_name) -> object:
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    # Spawn the vehicle in the environment
    def _set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        self.two_positions = [(9.530024528503418,302.57000732421875,-179.9996337890625),
            (193.77999877929688, 121.20999908447266 ,-89.99981689453125),
            (-7.529999732971191   ,288.2200012207031  ,89.99995422363281),
            (166.9145050048828   ,191.77003479003906  ,-0.00018310546875),
            (151.75006103515625   ,109.40003967285156  ,-0.00018310546875),
            (25.530019760131836   ,105.54998779296875  ,-179.9996337890625)]
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        #print(spawn_point)
        #position = self.two_positions[0]#random.choice(self.two_positions)
        #spawn_point = carla.Transform(carla.Location(x=position[0], y=position[1]), carla.Rotation(yaw= position[2]))
        #spawn_point.location.x = position[0]
        #spawn_point.location.y =  position[1]
        #spawn_point.rotation.yaw = position[2] 
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Setting destination for our vehicle
    # It's a random spawn point in that particular map

    def _set_destination(self, vehicle_location, spawn_points):
        destination = random.choice(spawn_points)
        while destination == vehicle_location:
            destination = random.choice(spawn_points)
        self.destination = destination

    # Clean up method
    def remove_sensors(self):
        if self.camera_obj is not None:
            del self.camera_obj
            self.camera_obj = None
        if self.collision_obj is not None:
            del self.collision_obj
            self.collision_obj = None
        if self.lane_invasion_obj is not None:
            del self.lane_invasion_obj
            self.lane_invasion_obj = None
        if VISUAL_DISPLAY:
            if self.env_camera_obj is not None:
                del self.env_camera_obj
                self.env_camera_obj = None
            else:
                self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None
        logging.debug("All the sensors have been removed.")


