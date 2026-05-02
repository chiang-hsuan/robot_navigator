import numpy as np
import math

def angle_wrap(angle: float) -> float:
    """
    Wrap any angle to the range [-pi, pi].

    Apply function avoids issues when angles jump from +pi to -pi (or vice versa).
    calculate angle using true side directly
    """
    wraped = np.arctan2(np.sin(angle), np.cos(angle)) # arctan2 nominator, denominator
    return wraped

def distance(p1,p2):
    dist = math.sqrt((p1[0][0]-p2[0][0])**2+(p1[1][0]-p2[1][0])**2)

    return dist


class TargetCourse:
    """
    Stores a 2D reference path and provides a very simple target-point search
    for Pure Pursuit.
    """

    def __init__(self, path: np.ndarray):
        """
        path: array of shape (N, 2) with columns [x, y]
        """
        self.px, self.py = path[:,0], path[:,1]
        
        
    def search_target_index(self, rear_x: float, rear_y: float, v: float, k: float, Lfc: float):
        """
        Find a target index on the path for Pure Pursuit.

        Very simple approach:
        1) Compute the lookahead distance Lf = k*v + Lfc
        2) Find the nearest path point to the rear axle position
        3) Starting from that nearest point, pick the first point whose distance
           from the rear axle is >= Lf (lookahead point)

        Returns:
            target_index (int): index into path arrays (cx, cy)
            Lf (float): lookahead distance actually used
        """
        # Lookahead distance:
        # - Lfc is a base lookahead distance (constant)
        # - k*v optionally increases lookahead with speed (k can be 0)
        lookahead_distance = Lfc + k * v

        # Distance from rear axle to every path point
        d = np.empty(len(self.px))
        #print(d.shape)
        for i, (x, y) in enumerate(zip(self.px,self.py)):
            d[i] = distance([rear_x,rear_y],[x,y])
        
        # Nearest point on the path
        nearest_point_ix = np.argmin(d)
        
        # Find the first point ahead (starting at nearest) that is at least Lf away
        # from the rear axle position.
        Lf = lookahead_distance
        lookahead_ix = np.where(d[nearest_point_ix:] >= Lf)[0]

        # If no point is far enough, target the last path point
        if len(lookahead_ix) == 0:
            lookahead_ix = len(self.px)-1
        else:
            lookahead_ix = lookahead_ix + nearest_point_ix # nearest point plus start from nearest point search

        return lookahead_ix, Lf#{"target_index":lookahead_ix,"Lf":Lf}
        


class PurePursuitNavigator:
    """
    Minimal Pure Pursuit controller:
    - Inputs: current vehicle state (x, y, yaw)
    - Output: constant speed command and steering command (delta)
    """

    def __init__(
        self,
        path: np.ndarray,
        wheelbase: float = 2.9,
        lookahead_dist: float = 2.0,
        speed: float = 2.0,
        k: float = 0.0,
        max_steer: float = math.pi / 4,
        goal_tolerance: float = 0.5,
    ):
        """
        Parameters:
            path: (N,2) array of [x,y] points
            wheelbase: vehicle wheelbase [m]
            lookahead_dist (Lfc): base lookahead distance [m]
            speed: commanded constant speed [m/s]
            k: speed gain for lookahead distance (Lf = k*v + Lfc)
            max_steer: steering limit [rad]
            goal_tolerance: stop when within this distance to final point [m]
        """
        self.path = path
        self.wheelbase = wheelbase
        self.speed = speed
        self.k = k
        self.Lfc = lookahead_dist
        self.max_steer = max_steer
        self.goal_tolerance = goal_tolerance

        self.course = TargetCourse(path)

    def navigate(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control commands using Pure Pursuit.

        state: column vector, at least 3x1: [x, y, yaw, ...]^T
               yaw is vehicle heading [rad]

        Returns:
            np.array([[v], [delta]])  -> speed [m/s], steering angle [rad]
        """
        
        # --- 1) Stop condition: if we are close to the final path point ---
        # print(self.path[-1], state[:2])
        if distance(self.path[-1],state[:2]) < self.goal_tolerance:
            return np.array([[0],[0]]) # speed, steering angle

        # --- 2) Choose speed (constant speed controller) ---
        

        # --- 3) Compute rear axle position ---
        # Many Pure Pursuit formulations use the rear axle as the reference point.
        # Here we approximate rear axle as "vehicle center minus half wheelbase".
        rear_axle = [state[0]-0.5*(self.wheelbase)*np.cos(state[2]), state[1]-0.5*self.wheelbase*np.sin(state[2])] # x-axis, y-axis
        
        # --- 4) Pick a target point on the path ---
        target_ix, Lf = self.course.search_target_index(rear_axle[0],rear_axle[1],self.speed,self.k,self.Lfc)
        #print(self.course.search_target_index(rear_axle[0],rear_axle[1],self.speed,self.k,self.Lfc))
        tx, ty = self.course.px[target_ix][0], self.course.py[target_ix][0]
        print(tx, ty)
        
        # --- 5) Compute heading error to the target point ---
        # alpha is the angle between vehicle heading and the line from rear axle to target.
        vector = [tx - rear_axle[0], ty - rear_axle[1]]
        alpha = np.arctan2(vector[1], vector[0]) - state[2] # back to heading
        alpha = angle_wrap(alpha)

        # --- 6) Pure Pursuit steering law ---
        # delta = atan2(2*L*sin(alpha), Lf)
        # where L is wheelbase and Lf is lookahead distance.
        delta = np.arctan2(2 * self.wheelbase * np.sin(alpha), Lf)
        
        # --- 7) Limit steering to feasible bounds ---
        delta = np.clip(delta,-self.max_steer,self.max_steer)

        #print(self.speed, delta)

        return np.array([[self.speed],delta]) # step data structure

