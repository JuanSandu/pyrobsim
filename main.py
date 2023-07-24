import pygame, random
import numpy as np


# Init the motor
pygame.init()

# Create the screen
sizeX = 800
sizeY = 600
univ_scale = 7.0 # pixel to meters ratio
clock = pygame.time.Clock()
dt = clock.tick(10) # TODO calculate from framerate
screen = pygame.display.set_mode( (sizeX, sizeY) )

# Title and icon
pygame.display.set_caption("Robot Sim Game")
# icon = pygame.image.load(img_path) # I don't have an icon for now
# pygame.display.set_icon(icon)


# Smoke
class Prop_Smoke:
    def __init__(self) -> None:
        # Prop smoke particles
        # [loc, vel, timer]
        self.smoke_left = []
        self.smoke_right = []
        self.rm_ptls = []
    
    def get_rot(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])

    def update_smoke(self, pos, angle, cmd, prop):
        # Left or right propulsor
        add_smoke = False
        if prop == "left":
            pos_d = [-16, 32] # Displacement
            smoke_ptls = self.smoke_left
            add_smoke = (cmd[0] > 0)
        elif prop == "right":
            pos_d = [16, 32] # Displacement
            smoke_ptls = self.smoke_right
            add_smoke = (cmd[1] > 0)

        if add_smoke:
            # Correct the position and vel of the prop
            rot_mat = self.get_rot(angle)
            prop_pos = pos + np.dot(pos_d, rot_mat)
            vel = np.dot([random.randint(5, 15)/10-1, 2], rot_mat)
            # Create new particles
            smoke_ptls.append([[int(prop_pos[0]), int(prop_pos[1])], 
                            vel, random.randint(4,6)])
        # Update their states
        rm_ptls = []
        for idx, particle in enumerate(smoke_ptls):
            particle[0][0] += particle[1][0]
            particle[0][1] += particle[1][1]
            particle[2] -= 0.1
            if particle[0][1] >= sizeY:
                particle[0][1] = sizeY
                particle[1][1] -= 0.5
            pygame.draw.circle(screen, (255, 255, 255), 
                            particle[0], 
                            particle[2])
            if particle[2] <= 0:
                rm_ptls.append(idx)
        # Remove timeout particles
        rm_ptls.reverse()
        for idx in rm_ptls:
            del smoke_ptls[idx]


# Propulsion
class Cold_Gas_Thruster:
    def __init__(self) -> None:
        self.prop_lc = 0.0
        self.min_prop = 0.0
        self.max_prop = 500.0

    def update_thrust(self, cmd):
        # Update
        self.prop_lc += cmd
        # Saturation
        self.prop_lc = max(self.min_prop,
                       min(self.max_prop, 
                           self.prop_lc) )


# Robot
class Lander:
    def __init__(self, init_state) -> None:
        """
        Initial position must be given in pixels for now.
        """
        # Lander representation
        self.landerImg = pygame.image.load("imgs/lander.png")
        self.robSizeX = 64
        self.robSizeY = 64
        # Components
        self.th_l = Cold_Gas_Thruster()
        self.th_r = Cold_Gas_Thruster()
        # Properties
        self.th_d = np.array([-0.5, 0.5]) # Ths. locations vs m.c.
        self.m = 50
        self.I = (1/2)*self.m*1.2 # Izz solid cilinder inertia
        self.g = -2#-9.8 # Moon gravity force
        self.G = np.array([0.0, self.g])
        # Simulation limits
        self.a_min = -30
        self.a_max = 30
        self.wd_min = -1.5
        self.wd_max = 1.5
        # State vector
        self.c_p = init_state[0:2]
        self.p = [self.c_p[0]/univ_scale,
                  (sizeY - self.c_p[1])/univ_scale]
        self.v = init_state[2:4]
        self.a = init_state[4:6]
        self.phi_deg = init_state[6]
        self.phi_rad = self.phi_deg * np.pi / 360
        self.w = init_state[7]
        self.wd = init_state[8]
        # Graphic effects
        self.smoke = Prop_Smoke()
        # Lander img
        self.img_rect = self.landerImg.get_rect(center=(self.c_p[0], 
                                                        self.c_p[1]))
    
    def sat_phi_rad(self):
        """
        Keeps phi between 0 and 2pi for it to be readable.
        """
        while -np.pi > self.phi_rad:
            self.phi_rad = self.phi_rad + 2*np.pi
        while self.phi_rad >= np.pi:
            self.phi_rad = self.phi_rad - 2*np.pi
    
    def thrust_lc2w(self, thrust):
        """
        Transforms (roation) thrust from local to world coordinates 
        frame.
        """
        self.phi_deg = self.phi_rad * 360 / (2*np.pi)
        # For now, rotation blocked
        # phi_rad = 0 # Uncomment to cancel rotations
        rot_mat = np.array([[np.cos(self.phi_rad), -np.sin(self.phi_rad)],
                            [np.sin(self.phi_rad),  np.cos(self.phi_rad)]])
        return np.dot( rot_mat, np.array([0.0, thrust]) )

    def dyn_model(self, cmd):
        """
        Runs the lander mathematical model to get the accelerations.
        """
        # Update thrusters
        self.th_l.update_thrust(cmd[0])
        self.th_r.update_thrust(cmd[1])
        # Local to world coordinates system
        self.prop_w = self.thrust_lc2w(self.th_l.prop_lc + 
                                       self.th_r.prop_lc)
        self.mom_w = np.dot(
                        np.array([self.th_l.prop_lc, 
                                  self.th_r.prop_lc]),
                        self.th_d )
        # Update accelerations
        self.a = self.prop_w / self.m + self.G
        self.wd = self.mom_w / self.I # Planar rotation
        # Saturate the values
        # TODO dead zones for avoiding static integration error
        self.a = np.clip(self.a, self.a_min, self.a_max)
        self.wd = np.clip(self.wd, self.wd_min, self.wd_max)
    
    def integrate_dyns(self):
        """
        Calculate new positions and velocities from accelerations.
        """
        # print(dt)
        self.v += np.multiply(self.a, np.array([dt, dt]))
        self.p += np.multiply(self.v, np.array([dt, dt]))
        self.w += np.multiply(self.wd, np.array([dt, dt]))[0]
        self.phi_rad += np.multiply(self.w, np.array([dt, dt]))[0]
        self.sat_phi_rad()
    
    def update_prop(self, cmd):
        """
        Updates the graphic effects.
        """
        self.smoke.update_smoke(self.c_p, self.phi_rad, cmd, "left")
        self.smoke.update_smoke(self.c_p, self.phi_rad, cmd, "right")

    def update_states(self, cmd):
        """
        Performs a simulation step.
        """
        self.dyn_model(cmd)
        self.integrate_dyns()
        self.update_prop(cmd)
    
    def sat_to_screen(self):
        """
        Keeps the robot within the screen limits
        """
        if (self.c_p[0] <= int(rob.robSizeX/2.0)): # Saturate min X
            self.c_p[0] = int(rob.robSizeX/2.0)
            rob.v[0] = 0
        elif (self.c_p[0] >= (sizeX-int(rob.robSizeX/2.0))): # Saturate max X
            self.c_p[0] = (sizeX-int(rob.robSizeX/2.0))
            rob.v[0] = 0
        if (self.c_p[1] <= int(rob.robSizeY/2.0)): # Saturate min Y
            self.c_p[1] = int(rob.robSizeY/2.0)
            rob.v[1] = 0
        elif (self.c_p[1] >= (sizeY-int(rob.robSizeY/2.0))): # Saturate max Y
            self.c_p[1] = (sizeY-int(rob.robSizeY/2.0))
            rob.v[1] = 0
    
    def rot_center(self, img, rect, angle, pos):
        """
        Rotates the img around the center and not the top left corner.
        """
        rot_img = pygame.transform.rotate(img, angle)
        diff_pos = pos-rect.center
        rot_rect = rot_img.get_rect(center=(rect.center+diff_pos))
        return rot_img, rot_rect
    
    def plot(self):
        """
        Corrected position, c_p, is used to change to pixel dimensions
        (display) from metric dimensions (simulation), p, where all the
        calculations happen.
        """
        # Limit its position to be inside the screen
        self.c_p[0] = rob.p[0] * univ_scale
        self.c_p[1] = (rob.p[1] - sizeY/univ_scale) * -univ_scale # Inverted Y
        self.sat_to_screen()

        # Rotate image using robot rotation
        rot_img, rot_rect = self.rot_center(self.landerImg, self.img_rect, 
                                            self.phi_deg, self.c_p)
        
        screen.blit( rot_img, rot_rect ) # To drop on the screen


if (__name__ == "__main__"):
    running = True
    display_data = False
    disp_cnt = 0
    secs_disp = 0.75

    ticks = 30
    rob = Lander(np.array([370.0, 100.0,        # Position
                           0.0, 0.0,            # Velocity
                           0.0, 0.0,            # Acceleration
                           0.0, 0.0, 0.0]))     # Pose
    th_cmd = np.array([0.0, 0.0])

    while running:
        # Background RGB color, always the first
        screen.fill( (0, 0, 0) )
        dt = 1.0 / clock.tick(30)

        # rob.update_prop()

        # Check all possible events within the game
        # for each loop
        for event in pygame.event.get():
            # To be able of closing the game window
            if event.type == pygame.QUIT:
                running = False
            
            # Actions depending on the keyboard
            # Key pressed!
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    th_cmd[0] = 5.0
                if event.key == pygame.K_RIGHT:
                    th_cmd[1] = 5.0

            # Key released!
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    th_cmd[0] = -10.0
                if event.key == pygame.K_RIGHT:
                    th_cmd[1] = -10.0
        
        # Simulation step
        rob.update_states(th_cmd)
        rob.plot()

        # Display data
        if display_data and (disp_cnt == int(secs_disp*ticks)):
            print("mmto: {}, ".format(rob.mom_w),
                  "phi: {}, w: {}, wd: {}".format(rob.phi_deg, rob.w, rob.wd),
                  "p: {}, v: {}, a: {}".format(rob.p, rob.v, rob.a))
            disp_cnt = 0
        else:
            disp_cnt += 1

        # Update the content of the window
        pygame.display.update()