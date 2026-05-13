#!/usr/bin/env python3

import rospy
import tf
from nav_msgs.msg import Odometry


class OdomTfBroadcaster(object):
    def __init__(self):
        self.parent_frame = rospy.get_param("~parent_frame", "")
        self.child_frame = rospy.get_param("~child_frame", "")
        self.broadcaster = tf.TransformBroadcaster()
        self.sub = rospy.Subscriber(
            "~odom", Odometry, self.odom_callback, queue_size=20, tcp_nodelay=True
        )

    def odom_callback(self, msg):
        parent = self.parent_frame or msg.header.frame_id
        child = self.child_frame or msg.child_frame_id

        if not parent or not child:
            rospy.logwarn_throttle(
                2.0, "Cannot publish odom TF because parent or child frame is empty."
            )
            return

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.broadcaster.sendTransform(
            (p.x, p.y, p.z),
            (q.x, q.y, q.z, q.w),
            stamp,
            child,
            parent,
        )


if __name__ == "__main__":
    rospy.init_node("odom_tf_broadcaster")
    OdomTfBroadcaster()
    rospy.spin()
