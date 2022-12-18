import os
import sys
import glob

try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    logging.error('Couldn\'t import Carla egg properly')

import carla
import logging
from simulation.settings import PORT, TIMEOUT, HOST, TOWN7

'''
try:
    
    Carla client library==0.9.12 installed in venv (virtual environment).No need to fetch Carla egg anymore!
    
    Make sure to to install all the dependencies provided with requirement.txt file.
    If not the following command can be be executed in the command line: 'pip install carla==0.9.12'
    
    
    import carla
    print("Here")
    logging.debug("Carla client library (0.9.12) is being imported...")

except ModuleNotFoundError:

    logging.critical(
        "Install/Import the carla client library first to connect to the server. Connection Failed!!!")
    sys.exit()

'''
class ClientConnection:
    def __init__(self):
        self.client = None

    def setup(self):
        try:

            # Connecting to the  Server
            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            self.world = self.client.load_world(TOWN7)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            return self.client, self.world

        except Exception as e:
            logging.error(
                'Failed to make a connection with the server: {}'.format(e))
            self.error_log()

    # An error log method: prints out the details if the client failed to make a connection
    def error_log(self):

        logging.debug("\nClient version: {}".format(
            self.client.get_client_version()))
        logging.debug("Server version: {}\n".format(
            self.client.get_server_version()))

        if self.client.get_client_version != self.client.get_server_version:
            logging.warning(
                "There is a Client and Server version mismatch! Please install or download the right versions.")
