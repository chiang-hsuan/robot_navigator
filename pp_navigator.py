import numpy as np
import math


def angle_wrap(angle: float) -> float:
    # angle clip to a range limit
    return angle

def distance(p1,p2):
    # compute distance between p1 and p2 points
    
    dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    return dist


class TargetCourse:
    # given path search a target point from current position

    def __init__(self, path: np.ndarray):
        # path: array of shape (N, 2) with columns [x, y]

        self.px, self.py = path[:,0], path[:,1]
        
        
    def search_target_index(self, rear_x: float, rear_y: float, v: float, k: float, Lfc: float):
        # Find a target index on the path for Pure Pursuit.

        '''
        parameters:
            Lfc: base lookahead distance
            k: hyperparameter controlling weight on speed influence to compute lookahead
        
        Returns:
            target_index (int): target index of path arrays (cx, cy)
            Lf (float): lookahead distance actually used
        '''
        # Lookahead distance:
        lookahead_distance = Lfc + k * v

        # Distance from rear axle to every path point
        d = np.array(len(self.x))
        for i, x, y in enumerate(zip(self.x,self.y)):
            d[i] = distance([rear_x,rear_y],[x,y])
        
        # Nearest point on the path
        nearest_point_ix = np.argmin(d)
        
        # Find the first point ahead (starting at nearest) that is at least Lf away
        # from the rear axle position.
        Lf = lookahead_distance
        lookahead_ix = np.where(d[nearest_point_ix:] >= Lf)[0]

        # If no point is far enough, target the last path point
        if lookahead_ix == None:
            lookahead_ix = len(self.px)-1
        else:
            lookahead_ix = lookahead_ix + nearest_point_ix # nearest point plus start from nearest point search

        return {"target_index":lookahead_ix,"Lf":Lf}
        


class PurePursuitNavigator:
    '''
    Minimal Pure Pursuit controller:
    Inputs: current vehicle state (x, y, yaw)
    Output: constant speed command and steering command (delta)
    '''

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
        '''
        Parameters:
            path: (N,2) array of [x,y] points
            wheelbase: vehicle wheelbase [m]
            lookahead_dist (Lfc): base lookahead distance [m]
            speed: commanded constant speed [m/s]
            k: speed gain for lookahead distance (Lf = k*v + Lfc)
            max_steer: steering limit [rad]
            goal_tolerance: stop when within this distance to final point [m]
        '''
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
        
        # Stop condition: if we are close to the final path point
        if distance(self.path[-1],state[:2]) < self.goal_tolerance:
            return np.array([0,0]) # speed, steering angle

        # Choose speed
        

        # Compute rear axle position
        # let the rear axle as the reference point.
        # approximate rear axle as "vehicle center minus half wheelbase".
        rear_axle = [state[0]-0.5*(self.wheelbase)*np.cos(state[2]), state[1]-0.5*self.wheelbase*np.sin(state[2])] # x-axis, y-axis

        
        # Pick a target point on the path
        target_ix, Lf = self.course.search_target_index(rear_axle[0],rear_axle[1],self.speed,self.k,self.Lfc)
        tx, ty = self.course.px[target_ix], self.course.py[target_ix]
        
        # Compute heading error to the target point
        # alpha is the angle between vehicle heading and the line from rear axle to target.
        vector = [tx - rear_axle[0], ty - rear_axle[1]]
        alpha = np.arccos(np.dot(vector,[1,0]),1,1) - state[2]
        alpha = angle_wrap(alpha)

        # Pure Pursuit steering law
        # delta = atan2(2*L*sin(alpha), Lf)
        # L is wheelbase and Lf is lookahead distance.
        delta = np.arctan2(2 * self.wheelbase * np.sin(alpha), Lf)
        
        # Limit steering to feasible bounds
        delta = np.clip(delta,-self.max_steer,self.max_steer)

        return np.array([self.speed,delta])
