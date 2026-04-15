import numpy as np
import math


def angle_wrap(angle: float) -> float:
    # angle clip to a range limit
    
    angle = np.clip(angle,[-np.pi,np.pi])
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
        nearest_distance = 10000
        nearest_point_ix = 0
        for i, x, y in enumerate(zip(self.x,self.y)):
            if distance([rear_x,rear_y],[x,y]) <= nearest_distance:
                nearest_point_ix = i
        
        # Find the first point ahead (starting at nearest) that is at least Lf away
        # from the rear axle position.
        Lf = lookahead_distance
        lookahead_idx = np.where(d[nearest_point_ix:] >= Lf)[0]

        # If no point is far enough, target the last path point
        Lf = [self.px[-1],self.py[-1]] if lookahead_idx == None

        return {"target_index":lookahead_idx,"Lf":Lf}
        


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
            break

        # Choose speed
        

        # Compute rear axle position
        # let the rear axle as the reference point.
        # approximate rear axle as "vehicle center minus half wheelbase".
        rear_axle = [state[0]-self.wheelbase/2,state[1]-self.wheelbase/2]
        
        
        # Pick a target point on the path
        target = self.course.search_target_index(rear_axle[0],rear_axle[1],self.speed,self.k,self.Lfc)
        
        # Compute heading error to the target point
        # alpha is the angle between vehicle heading and the line from rear axle to target.
        vector0 = [target[0] - rear_axle[0], target[1] - rear_axle[1]]
        alpha = np.arccos(np.dot(vector0,[1,0]),1,1) - state[2]

        # Pure Pursuit steering law
        # delta = atan2(2*L*sin(alpha), Lf)
        # L is wheelbase and Lf is lookahead distance.
        delta = np.arctan2(2 * self.wheelbase * np.sin(alpha), self.Lfc + self.k * self.speed)
        
        # --- 7) Limit steering to feasible bounds ---
        delta = angle_wrap(delta)

        return delta
