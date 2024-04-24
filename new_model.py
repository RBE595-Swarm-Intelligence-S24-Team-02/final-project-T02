import pygame
import numpy as np
import colorsys #HSV to RGB
import time
import math
from scipy.spatial import KDTree
from sympy import symbols, Eq, solve
from typing import List

# Constants
WHITE = (255, 255, 255)
TRAIL_COLOR = (255, 0, 255)
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 120 # Game FPS, not real-time FPS

# Variables to experiment with [VELOCITY_COUNT, NEIGHBOUR_DIST, ROBOT_COUNT] [36,200,4] [24,200,8] [12,400,16] [28,200,8] [12,200,24], [16,150,28]
END_SIMULATION = False
VELOCITY_COUNT = 36
NEIGHBOUR_DIST = 300
ROBOT_COUNT = 6
IS_VO = False
TIME_STEP = 0.1
ROBOT_RADIUS = 10
ROBOT_MAX_VELOCITY = 5

#SPAWN CIRCLE RADIUS
CIRCLE_SPAWN_RADIUS = 200

# MOST RECENT FILE TO COMBINE OUR WORK

class Robot:
    def __init__(self, radius, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color):
        self.current_location = np.array(spawn_location, dtype=float)
        self.velocity = initial_velocity
        self.goal = np.array(goal_location, dtype=float)
        self.radius = radius
        self.max_speed = max_speed
        self.time_step = time_step
        self.neighbours = []
        self.AV = []
        self.preferred_velocity()
        self.velocities = [np.zeros(2), self.pref_vel]
        self.trail = []  # Trail to store previous locations
        self.ID = id
        self.color = color

    def is_goal_reached(self):
        return np.linalg.norm(self.goal - self.current_location) < self.radius

    def update_current_location(self):
        self.trail.append(self.current_location.copy())  # Store the current location in the trail
        self.current_location += self.velocity * self.time_step

    def preferred_velocity(self):
        if not self.is_goal_reached():
            direction = self.goal - self.current_location
            direction_unit_vector = direction / np.linalg.norm(direction)
            self.pref_vel = self.max_speed * direction_unit_vector
        else:
            self.pref_vel = np.zeros(2)


    def calc_vel_penalty(self, robot2, new_vel, omega):
        """Calculate the penalty of a velocity
        
        # part of the penalty function is taken from
        # https://stackoverflow.com/questions/43577298/calculating-collision-times-between-two-circles-physics

        Args:
            robot2 (Robot): Neighbor robot
            new_vel (np.array): Velocity inside the RVO
            omega (int): Weight to change the impact of time to imapct on penalty

        Returns:
            int: Penalty value
        """
        
        new_vel = 2*new_vel - self.velocity
        distance = (self.radius + robot2.radius) ** 2

        # Calculate coefficients for a quadratic equation representing the time of collision
        a = (new_vel[0] - robot2.velocity[0]) ** 2 + (new_vel[1] - robot2.velocity[1]) ** 2

        b = 2 * ((self.current_location[0] - robot2.current_location[0]) * (new_vel[0] - robot2.velocity[0]) +
                 (self.current_location[1] - robot2.current_location[1]) * (new_vel[1] - robot2.velocity[1]))

        c = (self.current_location[0] - robot2.current_location[0]) ** 2 + (self.current_location[1] -
                                                                            robot2.current_location[1]) ** 2 - distance

        # Calculate the discriminant of the quadratic equation
        d = b ** 2 - 4 * a * c

        # Ignore glancing collisions that may not cause a response due to limited precision and lead to an infinite loop
        if b > -1e-6 or d <= 0:
            return np.linalg.norm(self.pref_vel - new_vel)

        # Calculate the square root of the discriminant
        e = math.sqrt(d)

        # Calculate the two potential times of collision (t1 and t2)
        t1 = (-b - e) / (2 * a)  # Collision time, +ve or -ve
        t2 = (-b + e) / (2 * a)  # Exit time, +ve or -ve

        # Check conditions to determine the actual collision time
        # If we are overlapping and moving closer, collide now
        if t1 < 0 < t2 and b <= -1e-6:
            # time_to_col = 0 #removed/commented-out because this would cause division by zero.
            return 0
        else:
            time_to_col = t1  # Return the time to collision
        penalty = omega * (1 / time_to_col) + np.linalg.norm(self.pref_vel - new_vel)

        return penalty


    def get_neighbours(self, kd_tree, all_robots, radius):
        self.neighbours = []
        neighbours_indices = kd_tree.query_ball_point(self.current_location, radius)
        for i in neighbours_indices:
            if all_robots[i] != self:
                self.neighbours.append(all_robots[i])

    def useable_velocities(self, combined_VO):
        self.AV = [vel for vel in self.velocities if tuple(vel) not in map(tuple, combined_VO)]

    def choose_velocity(self, combined_RVO, VO=False):
        self.preferred_velocity()
        
        VO = IS_VO
        
        if (VO):
            optimal_vel = self.pref_vel #ADDED FOR VO
            
        if not self.AV:
            min_penalty = float('inf')
            for neighbour in self.neighbours:
                for vel in combined_RVO:
                    penalty = self.calc_vel_penalty(neighbour, vel, omega=1)
                    if penalty < min_penalty:
                        min_penalty = penalty
                        # print('Robot ID: ', self.ID, 'Penalty: ', penalty, '\n')
                        optimal_vel = vel
        else:
            min_closeness = float('inf')
            for vel in self.AV:
                closeness = np.linalg.norm(self.pref_vel - vel)
                if closeness < min_closeness:
                    min_closeness = closeness
                    optimal_vel = vel

        if (VO):
            self.velocity = optimal_vel # VO part 
        else:
            self.velocity = (optimal_vel + self.velocity)/2 # RVO part (paper says this on page 3 definition 5)
        
        # if optimal_vel.any() != None:
        #     self.velocity = (optimal_vel + self.velocity)/2
        # else:
        #     self.velocity = self.velocity

    def compute_VO_and_RVO(self, robot2):

        VO = []
        RVO = []

        for vel in self.velocities:
            constraint_val = self.collision_cone_val(vel, robot2)
            if constraint_val < 0:
                VO.append(vel)
                RVO.append((vel + self.velocity) / 2)
        return VO, RVO

    def compute_combined_RVO(self, neighbour_robots):
        combined_VO = []
        combined_RVO = []
        for neighbour_robot in neighbour_robots:
            combined_VO.extend(self.compute_VO_and_RVO(neighbour_robot)[0])
            combined_RVO.extend(self.compute_VO_and_RVO(neighbour_robot)[1])
        return combined_VO, combined_RVO

    def collision_cone_val(self, vel, robot2):
        rx = self.current_location[0]
        ry = self.current_location[1]
        vrx = vel[0]
        vry = vel[1]

        obx = robot2.current_location[0]
        oby = robot2.current_location[1]
        vobx = robot2.velocity[0]
        voby = robot2.velocity[1]

        R = self.radius + robot2.radius + 10
        # if constraint_val >= 0, no collision , else there will be a collision in the future
        constraint_val = -((rx - obx) * (vrx - vobx) + (ry - oby) * (vry - voby)) ** 2 + (
                -R ** 2 + (rx - obx) ** 2 + (ry - oby) ** 2) * ((vrx - vobx) ** 2 + (vry - voby) ** 2)
        return constraint_val

    def draw(self, screen):
        # Draw the trail if there are at least two points
        if len(self.trail) >= 2:
            # pygame.draw.lines(screen, TRAIL_COLOR, False, self.trail, 2)
            pygame.draw.lines(screen, self.color, False, self.trail, 2)

        # Draw the robot as a circle
        pygame.draw.circle(screen, self.color, (int(self.current_location[0]), int(self.current_location[1])),
                           self.radius)

        # Draw a line representing the direction of the current velocity
        end_point = self.current_location + 10 * self.velocity
        pygame.draw.line(screen, (0, 0, 0), self.current_location, end_point, 2)
        
        # Draw Goal
        pygame.draw.circle(screen, self.color, (int(self.goal[0]), int(self.goal[1])),
                           self.radius, width=3)


def draw_robots(robots, screen):
    for robot in robots:
        robot.draw(screen)


def velocities_list():
    angles = np.linspace(0, 2*np.pi, num=VELOCITY_COUNT)
    x_values = np.cos(angles)
    y_values = np.sin(angles)
    for i in np.arange(0, 6, 1):
        if i>0:
            velocities = [i * np.array([x,y]) for x,y in zip(x_values,y_values)]
    return velocities


def generate_rainbow_colors(num_colors: int) -> List[tuple]:
    """
    Generates a list of RGB values for a rainbow with the specified number of colors.

    Args:
        num_colors: The number of colors in the rainbow.

    Returns:
        A list of RGB values.
    """

    if num_colors == 1:
        return [(255, 0, 0)]

    hues = [i / (num_colors) for i in range(num_colors)]
    rgbs = [
        tuple(
            [int(c * 255) for c in colorsys.hsv_to_rgb(h, 1.0, 1.0)]
        ) for h in hues
    ]
    return rgbs

def create_circular_locations(num_robots, radius, center):
    """
    This function creates two lists:
    - spawn_locations: list of spawn locations for each robot
    - goal_locations: list of goal locations for each robot
    around a circle with a specific center.

    Args:
        num_robots: integer, number of robots
        radius: integer, radius of the circle
        center: tuple, (x, y) coordinates of the circle's center

    Returns:
        spawn_locations: list of tuples representing spawn locations
        goal_locations: list of tuples representing goal locations
    """
    spawn_locations = []
    goal_locations = []
    theta = np.linspace(0, 2*np.pi, num_robots + 1)[:-1]
    
    for i, angle in enumerate(theta):
        x_offset = int(radius * np.cos(angle))
        y_offset = int(radius * np.sin(angle))
        x_spawn = center[0] + x_offset
        y_spawn = center[1] + y_offset
        spawn_locations.append((x_spawn, y_spawn))

        x_offset = int(radius * np.cos(angle + np.pi))
        y_offset = int(radius * np.sin(angle + np.pi))
        x_goal = center[0] + x_offset
        y_goal = center[1] + y_offset
        goal_locations.append((x_goal, y_goal))
        
    return spawn_locations, goal_locations

def create_robots(num_robots, radius):
    """
    This function creates a list of robots based on the number of robots needed.

    Args:
    num_robots: integer, number of robots
    radius: integer, radius of the circle
    rgb_list: list of tuples representing robot colors

    Returns:
    robots: list of Robot objects
    """
    robots = []
    rgb_list = generate_rainbow_colors(num_robots)
    center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    
    for i in range(num_robots):
        robot_name = "Robot{}".format(i+1)
        # radius, max_speed, spawn_location, goal_location, time_step, initial_velocity, id, color
        robot = Robot(
            radius=ROBOT_RADIUS,
            max_speed=ROBOT_MAX_VELOCITY,
            spawn_location=create_circular_locations(num_robots, radius, center)[0][i],
            goal_location=create_circular_locations(num_robots, radius, center)[1][i],
            time_step=TIME_STEP,
            initial_velocity=np.array([0, 0]),
            id=robot_name,
            color=rgb_list[i]
        )
        robots.append(robot)

    return robots


def update_time_counter(screen, start_time):
    # Update the time counter
    elapsed_time = time.time() - start_time
    # Add a black rectangle for the time counter
    font = pygame.font.Font(None, 24)
    # text = font.render(f"Time: {time.time():.2f}", True, (0, 0, 0))
    text = font.render(f"Time Step: {TIME_STEP}, Game FPS: {FPS}, Time: {elapsed_time:.2f} s", True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.topleft = (10, 10)
    screen.blit(text, text_rect)
    return elapsed_time


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Simulation")
    clock = pygame.time.Clock()
    
    # Start the timer
    start_time = time.time()

    # # Create robots with initial velocity
    # Diagonal Robots
    # robot1 = Robot(radius=10, max_speed=5, spawn_location=(50, 50), goal_location=(250, 250), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=1, color=(255,0,0))
    # robot2 = Robot(radius=10, max_speed=5, spawn_location=(250, 250), goal_location=(50, 50), time_step=TIME_STEP,
    #                initial_velocity=np.array([0, 0]), id=2, color=(0,0,255))
    # robots = [robot1, robot2]
    
    # Create robots - spawn in circle
    num_robots = ROBOT_COUNT          # number of robots
    spawn_radius = CIRCLE_SPAWN_RADIUS      # radius of spawning circle #was 200 with screen (600,600)
    robots = create_robots(num_robots, spawn_radius)

    # Compute velocities list for all robots
    velocities = velocities_list()

    for robot in robots:
        robot.velocities += velocities

    running = True
    time_elapsed_shown = False

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        count = 0
        for robot in robots:
            if robot.is_goal_reached():
                count += 1
        if count == len(robots):
            if not time_elapsed_shown:
                print(f"Time Elapsed: {elapsed_time:.2f} [s]")
                time_elapsed_shown = True
            if END_SIMULATION:
                running = False

        # Update KDTree
        kd_tree = KDTree([robot.current_location for robot in robots])
        combined_vos = []
        combined_rvos = []
        for robot in robots:
            robot.get_neighbours(kd_tree, robots, radius=NEIGHBOUR_DIST) #increased from 100
            vo, rvo = robot.compute_combined_RVO(robot.neighbours)
            combined_vos.append(vo)
            combined_rvos.append(rvo)

        # Update robots
        i = 0
        for robot in robots:
            robot.useable_velocities(combined_vos[i])
            robot.choose_velocity(combined_rvos[i])
            robot.update_current_location()
            i += 1

        # Draw on the screen
        screen.fill(WHITE)
        draw_robots(robots, screen)
        elapsed_time = update_time_counter(screen, start_time)
        
        

        pygame.display.flip()
        clock.tick(FPS)

    # print("Start Time:",start_time)
    print(f"Time Elapsed: {elapsed_time:.3f} [s]")
    
    pygame.display.quit()
    pygame.quit()


main()
