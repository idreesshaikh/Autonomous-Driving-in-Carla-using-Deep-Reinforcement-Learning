import math
import numpy as np
import weakref
import logging
import pygame
from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA


# ---------------------------------------------------------------------|
# ------------------------------- CAMERA |
# ---------------------------------------------------------------------|

class CameraSensor():

    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'128')
        front_camera_bp.set_attribute('image_size_y', f'128')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)#/255.0)



# ---------------------------------------------------------------------|
# ------------------------------- ENV CAMERA |
# ---------------------------------------------------------------------|

class CameraSensorEnv:

    def __init__(self, vehicle):

        pygame.init()
        self.display = pygame.display.set_mode((680,680),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    # Third camera is setup and provide the visual observations for our environment.

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'680')
        thrid_person_camera_bp.set_attribute('image_size_y', f'680')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        logging.info("Environment Camera has been setup.")
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()



# ---------------------------------------------------------------------|
# ------------------------------- COLLISION SENSOR|
# ---------------------------------------------------------------------|

# It's an important as it helps us to tract collisions
# It also helps with resetting the vehicle after detecting any collisions
class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)


# ---------------------------------------------------------------------|
# ------------------------------- OBSTACLE SENSOR|
# ---------------------------------------------------------------------|
'''
# It's an important as it helps us to detect Obstacles
class ObstacleSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.obstacle'
        self.parent = vehicle
        self.distance = None
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: ObstacleSensor._on_detection(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_obstacle_sensor(self, world) -> object:
        obstacle_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform()
        obstacle_sensor_bp.set_attribute('distance', f'10')
        obstacle_sensor_bp.set_attribute('only_dynamics', f'True')
        obstacle_sensor = world.spawn_actor(obstacle_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return obstacle_sensor

    @staticmethod
    def _on_detection(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.distance = event.distance

'''
# ---------------------------------------------------------------------|
# ------------------------------- LANE INVASION |
# ---------------------------------------------------------------------|

# Lane invasion sensor is a very important sensor!
# It's necessary for out vehicle not to deviate from the road


class LaneInvasionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.lane_invasion'
        self.parent = vehicle
        self.wrong_maneuver = False
        world = self.parent.get_world()
        self.sensor = self._set_lane_invasion_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_detection(weak_self, event))

    def _set_lane_invasion_sensor(self, world) -> object:
        lane_invasion_bp = world.get_blueprint_library().find(self.sensor_name)
        lane_invasion_sensor = world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.parent)
        return lane_invasion_sensor

    @staticmethod
    def _on_detection(weak_self, event):
        self = weak_self()
        if not self:
            return
        for x in event.crossed_lane_markings:
            #if x.lane_change == False:
            #    self.wrong_maneuver = True
            #elif (x.color == carla.LaneMarkingColor.Yellow and x.type == carla.LaneMarkingType.Solid):
            #    self.wrong_maneuver = True
            #elif (x.color == carla.LaneMarkingColor.Yellow and x.type == carla.LaneMarkingType.Broken):
            #    self.wrong_maneuver = True
            #elif(x.color == carla.LaneMarkingColor.Yellow and x.type == carla.LaneMarkingType.SolidSolid):
            #    self.wrong_maneuver = True
            #elif x.color == carla.LaneMarkingColor.White and x.type == carla.LaneMarkingType.Solid:
            #    self.wrong_maneuver = True
            if x.type == carla.LaneMarkingType.NONE:
                self.wrong_maneuver = True
            elif (x.type == carla.LaneMarkingType.Other):
                self.wrong_maneuver = True
            elif (x.type == carla.LaneMarkingType.Grass):
                self.wrong_maneuver = True
            elif (x.type == carla.LaneMarkingType.Curb):
                self.wrong_maneuver = True
            else:
                self.wrong_maneuver = False