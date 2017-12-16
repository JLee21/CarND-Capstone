#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from styx_msgs.msg import TrafficLightArray, TrafficLight, TrafficLightState, TrafficLightWaypoint
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
from operator import itemgetter
# record training images
import time
from time import sleep
import uuid
import os
import shutil

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    dl = lambda self, a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

    # Details from src/styx_msgs/msg/TrafficLight.msg
    state_txt = { TrafficLightState.RED: 'RED',
                  TrafficLightState.YELLOW: 'YELLOW',
                  TrafficLightState.GREEN: 'GREEN',
                  TrafficLightState.UNKNOWN: 'UNKNOWN' }
    labels_names = ['GREEN', 'RED', 'UNKNOWN','YELLOW']

    def __init__(self):

        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.light_waypoints = []
        self.light_visible = False

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLightState.UNKNOWN
        self.last_state = TrafficLightState.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.waypoints = None
        self.waypoints_count = 0
        self.rot = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # get config values for site
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        # enable/disable image recording for the use of training data
        if self.config['data_record_flag']:
            # folder housekeeping
            base_dir = 'training'

            if os.path.exists(os.path.abspath(base_dir)):
                shutil.rmtree(os.path.abspath((base_dir)))

            for _, light_state in self.state_txt.items():
                light_dir = os.path.join(base_dir, light_state)
                os.makedirs(os.path.abspath(light_dir))
            # subscribe to image topic
            sub7 = rospy.Subscriber('/image_color', Image, self.record_training_data_callback, queue_size=1)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', TrafficLightWaypoint, queue_size=1)

        # Calculate the distance from the traffic light (in waypoints) before starting to detect
        # the traffic light state.
        speed_limit = rospy.get_param('/waypoint_loader/velocity')
        self.light_wp_distance = 200 if speed_limit == 40 else 15

        rospy.spin()

    def record_training_data_callback(self, msg):
        sec_to_sleep = 0.7
        light_wp, line_wp, state = self.process_traffic_lights()
        # print('[record_training_data_callback] state: ', state)
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            resize_image = cv2.resize(cv2_img, (277, 303))
            cropped_image = resize_image[26:303, 0:277]
        except CvBridgeError, e:
            print(e)
        else:
            if state == 0:
                # cv2.imwrite('training/red/red_image_raw%s.jpeg' % (str(uuid.uuid4())), cv2_img)
                # cv2.imwrite('training/red/red_image_resize%s.jpeg' % (str(uuid.uuid4())), resize_image)
                cv2.imwrite('training/RED/red_image_crop_%s.jpeg' % (str(int(time.time()*100))), cropped_image)
                print('{0:10} image written at time {1}'.format('RED', time.strftime('%H:%M:%S')))
                sleep(sec_to_sleep)
            if state == 1:
                # cv2.imwrite('training/yellow/yellow_image_raw%s.jpeg' % (str(uuid.uuid4())), cv2_img)
                # cv2.imwrite('training/yellow/yellow_image_resize%s.jpeg' % (str(uuid.uuid4())), resize_image)
                cv2.imwrite('training/YELLOW/yellow_image_crop_%s.jpeg' % (str(int(time.time()*100))), cropped_image)
                print('{0:10} image written at time {1}'.format('YELLOW', time.strftime('%H:%M:%S')))
                sleep(sec_to_sleep)
            if state == 2:
                # cv2.imwrite('training/green/green_image_raw%s.jpeg' % (str(uuid.uuid4())), cv2_img)
                # cv2.imwrite('training/green/green_image_resize%s.jpeg' % (str(uuid.uuid4())), resize_image)
                cv2.imwrite('training/GREEN/green_image_crop_%s.jpeg' % (str(int(time.time()*100))), cropped_image)
                print('{0:10} image written at time {1}'.format('GREEN', time.strftime('%H:%M:%S')))
                sleep(sec_to_sleep)
            if state == 4:
                # cv2.imwrite('training/unknown/unknown_image_raw%s.jpeg' % (str(uuid.uuid4())), cv2_img)
                # cv2.imwrite('training/unknown/unknown_image_resize%s.jpeg' % (str(uuid.uuid4())), resize_image)
                cv2.imwrite('training/UNKNOWN/unknown_image_crop_%s.jpeg' % (str(int(time.time()*100))), cropped_image)
                print('{0:10} image written at time {1}'.format('UNKNOWN', time.strftime('%H:%M:%S')))
                sleep(sec_to_sleep)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # /base_waypoints should only be passed once at initialisation
        if self.waypoints:
            rospy.logerr("/base_waypoints message received multiple times!")
        self.waypoints = waypoints.waypoints
        self.waypoints_count = len(self.waypoints)
        rospy.logwarn("{:,} waypoints received from /base_waypoints.".format(self.waypoints_count))

    def traffic_cb(self, msg):
        """
        header:
          seq: 0
          stamp:
            secs: 1512922907
            nsecs: 189995050
          frame_id: /world
        pose:
          header:
            seq: 0
            stamp:
              secs: 1512922907
              nsecs: 190005064
            frame_id: /world
          pose:
            position:
              x: 363.378
              y: 1553.731
              z: 5.606708
            orientation:
              x: 0.0
              y: 0.0
              z: -0.010369102606
              w: 0.99994623941
        state:
          state: 0)
        """
        # print(['traffic_cb'], msg)
        self.lights = msg.lights
        if self.light_waypoints==[] and self.waypoints:
            # Create waypoints for the traffic signals and also the associated stop lines.
            self.light_waypoints = [self.get_closest_waypoint(light.pose.pose) for light in self.lights]
            rospy.logwarn("traffic light waypoints calculated as {}.".format(self.light_waypoints))
            self.stopline_waypoints = [self.get_closest_waypoint(Pose(Point(x,y,0.0),Quaternion(0.0,0.0,0.0,0.0))) for (x, y) in self.config['stop_line_positions']]
            rospy.logwarn("traffic light stopline waypoints calculated as {}.".format(self.stopline_waypoints))
            # Determine if the waypoints increase or decrease around the track.
            self.rot = 1 if self.light_waypoints[0]>self.stopline_waypoints[0] else -1

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, line_wp, state = self.process_traffic_lights()

        if light_wp != -1:
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                if self.last_state != self.state:
                    rospy.logwarn("traffic light: {}.".format(self.state_txt[self.state]))
                self.last_state = self.state
                light_wp = light_wp if state in (TrafficLightState.RED, TrafficLightState.YELLOW) else -1
                line_wp = line_wp if state in (TrafficLightState.RED, TrafficLightState.YELLOW) else -1
                self.last_wp = line_wp
                self.upcoming_red_light_pub.publish(TrafficLightWaypoint(line_wp, TrafficLightState(self.state)))
            else:
                self.upcoming_red_light_pub.publish(TrafficLightWaypoint(self.last_wp, TrafficLightState(self.state)))
            self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # Process if the waypoints list exists (from /base_waypoints)
        if self.waypoints:
            distances = [self.dl(pose.position, waypoint.pose.pose.position) for waypoint in self.waypoints]
            return min(enumerate(distances), key=itemgetter(1))[0]
        else:
            return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO: Temporary data from simulator until image processing is complete.
        #return self.lights[light].state.state

        if(not self.has_image):
            self.prev_light_loc = None
            rospy.logwarn('no image')
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
        # somtimes, car_position DNE
        else:
            return -1, -1, TrafficLightState.UNKNOWN


        # find the closest visible traffic light (if one exists)
        if self.rot==1:
            distances = [tl - car_position if tl>car_position else len(self.waypoints) + tl - car_position for tl in self.stopline_waypoints]
        else:
            distances = [car_position - tl if tl<car_position else len(self.waypoints) + car_position - tl for tl in self.stopline_waypoints]
        lights = [waypoint for d, waypoint in zip(distances,self.light_waypoints) if d < self.light_wp_distance]
        light_waypoint = None if len(lights)==0 else lights[0]

        if light_waypoint:
            if light_waypoint in self.light_waypoints:
                light = self.light_waypoints.index(light_waypoint)
                stopline_waypoint = self.stopline_waypoints[light]
                state = self.get_light_state(light)
                if not self.light_visible:
                    rospy.logwarn("traffic light approaching - {}.".format(self.state_txt[state]))
                    self.light_visible = True
                return light_waypoint, stopline_waypoint, state
        if self.light_visible:
            rospy.logwarn("traffic light passed.")
            self.light_visible = False
        return -1, -1, TrafficLightState.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
