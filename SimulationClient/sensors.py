import math
import numpy as np
import weakref
import cv2
import logging
from SimulationClient.connection import carla
from SimulationClient.settings import RGB_CAMERA, SSC_CAMERA


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
        front_camera_bp.set_attribute('fov', f'120')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=0.8)), attach_to=self.parent)
        logging.info("Front Camera has been setup on the vehicle.")
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        self.front_camera.append(placeholder2/255.0)
        #cv2.imshow("Front Camera", placeholder2)
        #cv2.waitKey(1)


class CameraSensorEnv:

    def __init__(self, vehicle):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        #self.third_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

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
        # image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        cv2.imshow("Front Camera", placeholder2)
        cv2.waitKey(1)

# ---------------------------------------------------------------------|
# ------------------------------- LIDAR |
# ---------------------------------------------------------------------|

"""
class LiDarSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.lidar.ray_cast_semantic'
        self.parent = vehicle
        self.data = list()
        world = self.parent.get_world()
        self.sensor = self._set_lidar_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LiDarSensor._get_lidar_sensor_data(weak_self, event))

    # LiDar sensor is setup to get the surrounding readings in order to make better predictions.
    def _set_lidar_sensor(self, world) -> object:
        lidar_bp = world.get_blueprint_library().find(self.sensor_name)
        lidar_bp.set_attribute('range', f'3.0')
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.5, z=1.0))
        lidar = world.spawn_actor(
            lidar_bp, sensor_relative_transform, attach_to=self.parent)
        logging.info("Lidar has been setup on the vehicle.")
        return lidar

    @staticmethod
    def _get_lidar_sensor_data(weak_self, image) -> None:
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))

        x = np.array(points[:32, 0]).ravel()
        y = np.array(points[:32, 1]).ravel()
        z = np.array(points[:32, 2]).ravel()
        self.data.append(np.array([x, y, z]).ravel())

"""
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
        logging.info("Collision sensor has been setup on the vehicle.")
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        logging.debug("Vehicle has collided!")
        self.collision_data.append(intensity)


# ---------------------------------------------------------------------|
# ------------------------------- RADAR |
# ---------------------------------------------------------------------|

'''
# Radar sensor is setup to get the surrounding readings in order to make better predictions.

class RadarSensor:
    def __init__(self, vehicle):
        self.sensor_name = 'sensor.other.radar'
        self.parent = vehicle
        self.radar_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_radar_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda data: RadarSensor._get_radar_sensor_data(weak_self, data))

    def _set_radar_sensor(self, world):
        radar_bp = world.get_blueprint_library().find(self.sensor_name)
        radar_bp.set_attribute('horizontal_fov', f'35')
        radar_bp.set_attribute('vertical_fov', f'20')
        radar = world.spawn_actor(radar_bp, carla.Transform(
            carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=8.0)), attach_to=self.parent)
        logging.info("Radar sensor has been setup on the vehicle.")
        return radar

    @staticmethod
    def _get_radar_sensor_data(weak_self, data) -> None:
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(data), 4))

        d = np.mean(points[:, 2])
        v = np.mean(points[:, 3])

        self.radar_data.append(np.array([d, v], dtype=np.float32).ravel())
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
        logging.info("Lane invasion sensor has been setup on the vehicle.")
        return lane_invasion_sensor

    @staticmethod
    def _on_detection(weak_self, event):
        self = weak_self()
        if not self:
            return
        for x in event.crossed_lane_markings:
            # if x.lane_change == False:
            #    self.wrong_maneuver = True
            #    logging.debug("X( Wrong turn!")
            if (x.color == carla.LaneMarkingColor.Yellow and x.type == carla.LaneMarkingType.Solid):
                self.wrong_maneuver = True
                logging.debug("X( Wrong turn!")
            #elif (x.color == carla.LaneMarkingColor.Yellow and x.type == carla.LaneMarkingType.Broken):
                #self.wrong_maneuver = True
                #logging.debug("X( Wrong turn!")
            elif(x.color == carla.LaneMarkingColor.Yellow and x.type == carla.LaneMarkingType.SolidSolid):
                self.wrong_maneuver = True
                logging.debug("X( Wrong turn!")
            elif x.color == carla.LaneMarkingColor.White and x.type == carla.LaneMarkingType.Solid:
                self.wrong_maneuver = True
                logging.debug("X( Wrong turn!")
            elif x.type == carla.LaneMarkingType.NONE:
                self.wrong_maneuver = True
                logging.debug("X( Wrong turn!")
            else:
                self.wrong_maneuver = False
