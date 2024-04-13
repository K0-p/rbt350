import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  t = 1-np.cos(angle)
  c = np.cos(angle)
  s = np.sin(angle)
  x, y, z = axis

  rot_mat = np.array([[t*x*x+c, t*x*y-s*z, t*x*z+s*y],
                      [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
                      [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """
  rot_matrix = rotation_matrix(axis, angle)
  T = np.block([[rot_matrix, np.array([v_A]).T],
                [0, 0, 0, 1]])
  
  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """

  hip_frame = homogenous_transformation_matrix([0, 0, 1], joint_angles[0], [0, 0, 0])
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  hipToShoulder = homogenous_transformation_matrix([0, 1, 0], joint_angles[1], [0, -1*HIP_OFFSET, 0])
  shoulder_frame = np.matmul(fk_hip(joint_angles), hipToShoulder)
  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

of the elbow frame in the base frame
  """

  # remove these lines when you write your solution
  shoulderToElbow = homogenous_transformation_matrix([0, 1, 0], joint_angles[2], [0, 0, UPPER_LEG_OFFSET])
  elbow_frame = np.matmul(fk_shoulder(joint_angles), shoulderToElbow)
  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  elbowToFoot = homogenous_transformation_matrix([1, 1, 1], 0, [0, 0, LOWER_LEG_OFFSET])
  end_effector_frame = np.matmul(fk_elbow(joint_angles), elbowToFoot)
  return end_effector_frame
