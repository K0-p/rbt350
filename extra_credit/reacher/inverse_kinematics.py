import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 10

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    # Initialize cost to zero
    cost = 0.0

    # Add your solution here.
    # Calculate the end-effector position of the guess
    guess_xyz = forward_kinematics.fk_foot(guess)[:3, 3]
    #print('guess_xyz:',guess_xyz)

    # Calculate the cost based on the guess and end-effector position
    difference = np.subtract(guess_xyz, end_effector_pos)
    cost = np.sqrt(np.sum(np.square(difference)))
    #print('cost:',cost)
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """

    # Initialize Jacobian to zero
    J = np.zeros((3, 3))
    # Add your solution here.
    # delta_theta = np.array([delta, delta, delta]).transpose()
    # perturbed_xyz = forward_kinematics.fk_foot(np.add(joint_angles, delta_theta))[:3, 3]
    # delta_xyz = np.subtract(perturbed_xyz, forward_kinematics.fk_foot(joint_angles)[:3, 3])
    # print(delta_xyz)
    # print(delta_theta)
    # print(np.linalg.pinv(delta_theta))
    # J[:, i] = delta_xyz / (2 * delta) 

    for i in range(3):
        curr_angles = joint_angles
        curr_angles[i] += delta
        forward_xyz = forward_kinematics.fk_foot(curr_angles)[:3, 3]
        curr_angles[i] -= 2 * delta
        back_xyz = forward_kinematics.fk_foot(curr_angles)[:3, 3]
        J[:, i] = (forward_xyz - back_xyz) / (2 * delta)

    # for i in range(3):
    #     delta_theta = np.zeros(3)
    #     delta_theta[i] = delta
    #     perturbed_xyz_pos = forward_kinematics.fk_foot(np.add(joint_angles, delta_theta))[:3, 3]
    #     perturbed_xyz_neg = forward_kinematics.fk_foot(np.subtract(joint_angles, delta_theta))[:3, 3]
    #     delta_xyz = np.subtract(perturbed_xyz_pos, perturbed_xyz_neg)
    #     J[:, i] = delta_xyz / (2 * delta)  # Finite difference approximation

    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    # Initialize previous cost to infinity
    previous_cost = np.inf
    # Initialize the current cost to 0.0
    cost = 0.0

    for iters in range(MAX_ITERATIONS):
        # Calculate the Jacobian matrix using finite differences
        J = calculate_jacobian_FD(guess, PERTURBATION)

        # Calculate the residual
        residual = np.subtract(end_effector_pos, forward_kinematics.fk_foot(guess)[:3, 3])
    
        # Compute the step to update the joint angles using the Moore-Penrose pseudoinverse using numpy.linalg.pinv
        delta_theta = np.matmul(np.linalg.pinv(J), residual)

        # Take a full Newton step to update the guess for joint angles
        guess = np.add(guess, delta_theta)

        # cost = # Add your solution here.
        cost = ik_cost(end_effector_pos, guess)

        # Calculate the cost based on the updated guess
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost

    return guess

# print(calculate_jacobian_FD(np.array([0, 0, 0.5]), 0.0001))
