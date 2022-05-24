import math
import time
import random
import numpy as np
import logging
from SimulationClient.connection import carla, logging
from SimulationClient.settings import CAR_NAME, EPISODE_LENGTH, NUMBER_OF_VEHICLES, NUMBER_OF_PEDESTRIAN
from SimulationClient.sensors import CameraSensor, CameraSensorEnv, CollisionSensor, LaneInvasionSensor


class CarlaEnvironment():

    def __init__(self, client, world) -> None:

        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.vehicle = None
        self.settings = None
        self.front_camera = None
        self.third_person_view = None
        #self.lidar_sensor = None
        #self.radar_sensor = None
        self.collision_history = None
        self.wrong_maneuver = None
        self.destination = None
        self.waypoint = None
        self.trajectory = None
        self.velocity = None
        self.location = None
        self.prev_steer = None
        self.signed_waypoint_distance = None
        self.continus_timestamp = 0
        self.action_space = self._get_action_space()
        self.episode_start_time = None

        # Objects to be kept alive
        self.camera_obj = None
        self.third_view_obj = None
        self.collision_obj = None
        #self.lidar_obj = None
        #self.radar_obj = None
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
            self.third_view_obj = CameraSensorEnv(self.vehicle)
            self.sensor_list.append(self.third_view_obj.sensor)

            # Quick start our vehicle from its initial state
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0))

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            """
            # Lidar sensor
            self.lidar_obj = LiDarSensor(self.vehicle)
            while(len(self.lidar_obj.data) == 0):
                time.sleep(0.001)
            self.lidar_sensor = self.lidar_obj.data.pop(-1)
            self.sensor_list.append(self.lidar_obj.sensor)
            """
            '''
            # Radar sensor
            self.radar_obj = RadarSensor(self.vehicle)
            while(len(self.radar_obj.radar_data) == 0):
                time.sleep(0.001)
            self.radar_sensor = self.radar_obj.radar_data.pop(-1)
            self.sensor_list.append(self.radar_obj.sensor)
            '''

            # Lane Invasion sensor
            self.lane_invasion_obj = LaneInvasionSensor(self.vehicle)
            self.wrong_maneuver = self.lane_invasion_obj.wrong_maneuver
            self.sensor_list.append(self.lane_invasion_obj.sensor)

            # Quick start our vehicle from its initial state
            #self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=1.0, steer=0.0))

            # Velocity
            velocity = self.vehicle.get_velocity()
            self.velocity = math.sqrt( velocity.x**2 + velocity.y**2 + velocity.z**2) / 10

            # Rotation
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()
            self.prev_steer = 0.0

            # Waypoint nearby angle and distance from it
            angles = list()
            distances = list()

            self.waypoint = self.map.get_waypoint(
                self.location, project_to_road=True, lane_type=(carla.LaneType.Driving))

            for i in range(5):
                self.waypoint = self.waypoint.next(1.0)[-1]
                distances.append(math.sqrt((self.waypoint.transform.location.x - self.location.x)** 2 + (self.waypoint.transform.location.y-self.location.y)**2))
                angles.append(abs(self.rotation) - abs(self.waypoint.transform.rotation.yaw))

            self.trajectory = np.mean(angles) / 10
            self.signed_waypoint_distance = np.mean(distances)
            if self.trajectory < 0.0:
                self.signed_waypoint_distance * -1

            self.collision_history.clear()
            self.continus_timestamp = 0

            # Time noted for the start of the episode
            self.episode_start_time = time.time()

            self.nav_data = np.array([self.trajectory, self.velocity, self.prev_steer, self.signed_waypoint_distance])

            logging.info("Environment has been resetted.")

            return self.front_camera, self.nav_data

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.


    def _step(self, action_index):
        try:
            
            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            
            
            # Action fron action space for contolling the vehicle with a discrete action
            #action("index just make")
            action = self.action_space[action_index]
            
            if self.velocity >= 25.0:
                self.vehicle.apply_control(carla.VehicleControl(steer=action[1]))
            else:
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
            self.velocity = self.velocity/36
            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            new_location = self.vehicle.get_location()

            # Location of the vehicle
            vehicle_x, vehicle_y = new_location.x, new_location.y
            distance_covered = math.sqrt((new_location.x - self.location.x)**2 + (new_location.y - self.location.y)**2)
            self.location = new_location

            # Randomly picked next waypoint in 1.0m distance.
            # Waypoint broken down in its three necessary components.
            # Waypoint nearby angle and distance from it
            angles = list()
            distances = list()

            self.waypoint = self.map.get_waypoint(
                self.location, project_to_road=True, lane_type=(carla.LaneType.Driving))

            for i in range(5):
                self.waypoint = self.waypoint.next(1.0)[-1]
                distances.append(math.sqrt((self.waypoint.transform.location.x - self.location.x)** 2 + (self.waypoint.transform.location.y-self.location.y)**2))
                angles.append(abs(self.rotation) - abs(self.waypoint.transform.rotation.yaw))

            self.trajectory = np.mean(angles) / 5
            self.signed_waypoint_distance = np.mean(distances)
            if self.trajectory < 0.0:
                self.signed_waypoint_distance * -1
            
            
            # Rewards are given below!
            

            if len(self.collision_history) != 0:
                done = True
                reward = -20 * self.collision_history.pop(-1)
                print("Vehicle has collided.")

            elif self.wrong_maneuver:
                done = True
                reward = -500 * (abs(self.location.x - self.waypoint.transform.location.x) + abs(self.location.y - self.waypoint.transform.location.y))
                print("Vehicle has gone out of the lane.")

            else:
                done = False
                if (5 * distance_covered)  <= 1.0:
                    reward = 5 * distance_covered + (self.velocity)
                else:
                    reward = 5 * distance_covered**2 + (self.velocity)

            if self.velocity * 36 <= 2.0:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.2))

            if self.episode_start_time + EPISODE_LENGTH < time.time():
                print("Times up.")
                reward = 500
                done = True

            elif (self.destination.location.x <= vehicle_x + 1.0 and self.destination.location.x >= vehicle_x - 1.0) and (self.destination.location.y <= vehicle_y + 1.0 and self.destination.location.y >= vehicle_y - 1.0):
                print("Reached the destination.")
                reward = 1000
                done = True
            
            # tick
            while (self.front_camera is None):
                time.sleep(0.001)

            # navigation comprises of 4 data points
            self.nav_data = np.array([self.trajectory, self.velocity, self.prev_steer, self.signed_waypoint_distance])

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
            self.world.set_pedestrians_cross_factor(0)
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
                bp_vehicle = random.choice(
                    self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            logging.info("NPC vehicles have been generated in autopilot mode.")
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
        steer = np.array([-0.75, -0.50, -0.25 -0.1, 0.0, 0.1, 0.25, 0.5, 0.75])
        accelerate = np.array([0.0, 0.50])

        action_space = \
            np.array([
                [0.0, -0.75],
                [0.0, -0.50],                
                [0.0, -0.25],
                [0.0,  -0.1],
                [0.0,   0.0],
                [0.0,   0.1],
                [0.0,   0.25],
                [0.0,   0.50],
                [0.0,   0.75],
                
                [0.5,  -0.75],
                [0.5,  -0.50],
                [0.5,  -0.25],
                [0.5,   -0.1],
                [0.5,    0.0],
                [0.5,    0.1],
                [0.5,   0.25],
                [0.5,   0.50],
                [0.5,   0.75],
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
        spawn_point = random.choice(
            spawn_points) if spawn_points else carla.Transform()
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
        if self.third_view_obj is not None:
            del self.third_view_obj
            self.third_view_obj = None
        if self.collision_obj is not None:
            del self.collision_obj
            self.collision_obj = None
        #if self.lidar_obj is not None:
            #del self.lidar_obj
            #self.lidar_obj = None
            #self.radar_obj = None
        if self.lane_invasion_obj is not None:
            del self.lane_invasion_obj
            self.lane_invasion_obj = None
        self.front_camera = None
        self.third_person_view = None
        #self.lidar_sensor = None
        self.collision_history = None
        self.wrong_maneuver = None
        logging.debug("All the sensors have been removed.")
