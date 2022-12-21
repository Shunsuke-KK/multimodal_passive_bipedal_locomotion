import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import random
import mujoco_py
import os

class PassiveBipedal_2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    m_hip = 20
    m_thigh = 6
    m_tibia = 3
    m_foot = 1
    k1 = 6000
    k2 = 6000
    k3 = 10000
    k_waste = 25
    g = 9.8
    thigh_x = 0.1
    thigh_z = 0.4
    r_hip = 0.05
    l1 = 0.4
    l2 = 0.2
    l3 = 0.2
    geer =800

    def __init__(self,path):
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, "passivebipedal.xml"), 4)
        utils.EzPickle.__init__(self,)
        
    def energy_measure(self):
        m_hip = self.m_hip
        m_thigh = self.m_thigh
        m_tibia = self.m_tibia
        m_foot = self.m_foot
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k_waste = self.k_waste
        g = self.g
        thigh_x = self.thigh_x
        thigh_z = self.thigh_z
        r_hip = self.r_hip
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3

        # energy
        #kinematic(tlanslational & rotation)
        t  = 0.5*m_hip*(np.square(self.sim.data.get_site_xvelp("s_hip")).sum())
        t += 0.5*m_thigh*(np.square(self.sim.data.get_site_xvelp("s_waste")).sum())
        t += 0.5*m_thigh*(np.square(self.sim.data.get_site_xvelp("s_left_waste")).sum())
        t += 0.5*m_tibia*(np.square(self.sim.data.get_site_xvelp("s_tibia")).sum())
        t += 0.5*m_tibia*(np.square(self.sim.data.get_site_xvelp("s_left_tibia")).sum())
        t += 0.5*m_foot*(np.square(self.sim.data.get_site_xvelp("s_foot")).sum())
        t += 0.5*m_foot*(np.square(self.sim.data.get_site_xvelp("s_left_foot")).sum())

        I_hip = 0.5*m_hip*r_hip*r_hip
        if(self.sim.data.get_joint_qpos("waste_z")<0):
            I_thigh_right = m_thigh*(thigh_x*thigh_x + thigh_z*thigh_z)/12 
            + m_thigh*((thigh_z - abs(self.sim.data.get_joint_qpos("waste_z"))) / thigh_z)*((thigh_z/2 - abs(self.sim.data.get_joint_qpos("waste_z")))**2)
            + m_thigh*(abs(self.sim.data.get_joint_qpos("waste_z")) / thigh_z)*((self.sim.data.get_joint_qpos("waste_z"))**2)
        else:
            I_thigh_right = m_thigh*(thigh_x*thigh_x + thigh_z*thigh_z)/12 + m_thigh*((thigh_z/2 + self.sim.data.get_joint_qpos("waste_z"))**2)
        
        if(self.sim.data.get_joint_qpos("left_waste_z")<0):
            I_thigh_left = m_thigh*(thigh_x*thigh_x + thigh_z*thigh_z)/12 
            + m_thigh*((thigh_z - abs(self.sim.data.get_joint_qpos("left_waste_z"))) / thigh_z)*((thigh_z/2 - abs(self.sim.data.get_joint_qpos("left_waste_z")))**2)
            + m_thigh*(abs(self.sim.data.get_joint_qpos("left_waste_z")) / thigh_z)*((self.sim.data.get_joint_qpos("left_waste_z"))**2)
        else:
            I_thigh_left = m_thigh*(thigh_x*thigh_x + thigh_z*thigh_z)/12 + m_thigh*((thigh_z/2 + self.sim.data.get_joint_qpos("left_waste_z"))**2)
        
        r  = 0.5*I_hip*((self.sim.data.get_sensor("rooty_omega"))**2)
        r += 0.5*I_thigh_right*((self.sim.data.get_sensor("waste_omega"))**2 + (self.sim.data.get_sensor("rooty_omega"))**2)
        r += 0.5*I_thigh_left*((self.sim.data.get_sensor("left_waste_omega"))**2 + (self.sim.data.get_sensor("rooty_omega"))**2)

        #kinematic energy
        kinematic = t + r

        # elastic energy
        elastic  = 0.5*k1*((self.sim.data.get_sensor("1_right_length") - l1)**2)
        elastic += 0.5*k1*((self.sim.data.get_sensor("1_left_length")  - l1)**2)
        elastic += 0.5*k2*((self.sim.data.get_sensor("2_right_length") - l2)**2)
        elastic += 0.5*k2*((self.sim.data.get_sensor("2_left_length")  - l2)**2)
        elastic += 0.5*k3*((self.sim.data.get_sensor("3_right_length") - l3)**2)
        elastic += 0.5*k3*((self.sim.data.get_sensor("3_left_length")  - l3)**2)
        elastic += 0.5*k_waste*(self.sim.data.get_joint_qpos("waste_joint")**2)
        elastic += 0.5*k_waste*(self.sim.data.get_joint_qpos("left_waste_joint")**2)

        # potential energy
        height_std = 0.4
        potential  =  m_hip *g*(self.sim.data.get_site_xpos('s_hip')[2] - height_std)
        potential += m_thigh*g*(self.sim.data.get_site_xpos("s_thigh")[2] - height_std)
        potential += m_thigh*g*(self.sim.data.get_site_xpos("s_left_thigh")[2] - height_std)
        potential += m_tibia*g*(self.sim.data.get_site_xpos('s_tibia')[2] - height_std)
        potential += m_tibia*g*(self.sim.data.get_site_xpos('s_left_tibia')[2] - height_std)
        potential += m_foot *g*(self.sim.data.get_site_xpos('s_foot')[2] - height_std)
        potential += m_foot *g*(self.sim.data.get_site_xpos('s_left_foot')[2] - height_std)

        energy = kinematic + potential + elastic

        return energy,kinematic,potential,elastic

    def waste_pos(self):
        return np.array([self.sim.data.get_joint_qpos("waste_z"),self.sim.data.get_joint_qpos("left_waste_z")])
    
    def model_mass(self):
        mass_sum = self.m_hip + 2*self.m_thigh + 2*self.m_tibia + 2*self.m_foot
        return mass_sum

    def contact_force(self):
        right_x_N = 0
        left_x_N = 0
        right_z_N = 0
        left_z_N = 0

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            if(int(contact.geom2)<30):
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            if(int(contact.geom2)>30):
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
        return np.array([right_z_N,left_z_N])

    def contact(self):
        right_contact = 0
        left_contact = 0

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            if(int(contact.geom2)<30):
                right_contact = 1
            if(int(contact.geom2)>30):
                left_contact = 1
        return np.array([right_contact,left_contact])


    def step(self,a,a_before=np.zeros(2),num=0,w2=0,view=True):
        forward_coefficient = w2
        penalty = 1.0
        energy_coefficient = 0.4
        symmetry_coefficient = 0.15
        joint_coefficient = 0.3 #delete
        mini = 0.6

        if num<500:
            energy_coefficient = 0.15*energy_coefficient
            symmetry_coefficient = 0.*symmetry_coefficient

        alive_bonus = 1.0
        gear = self.geer
        pena = 0
        sym_flag = False

        if(len(a) ==4):
            a,a_symmetry = np.split(a,2)
            a_compare = np.zeros(2)
            a_compare[0] = a_symmetry[1]
            a_compare[1] = a_symmetry[0]
            sym_flag = True
            symmetry = np.square(a - a_compare).sum()

        # save the previous status & action
        posbefore = self.sim.data.get_joint_qpos("rootx")
        joint_before = abs(self.sim.data.get_joint_qpos("waste_joint") - self.sim.data.get_joint_qpos("left_waste_joint"))

        #energy
        energy_b,_,_,_ = self.energy_measure()

        ######################################
        self.do_simulation(a, self.frame_skip)
        ######################################

        posafter = self.sim.data.get_joint_qpos("rootx")
        ang = self.sim.data.get_joint_qpos("rooty")
        joint_after = abs(self.sim.data.get_joint_qpos("waste_joint") - self.sim.data.get_joint_qpos("left_waste_joint"))

        #energy
        energy_a,_,_,_ = self.energy_measure()

        #force
        force =  gear*abs(self.sim.data.get_sensor("waste_F"))
        force += gear*abs(self.sim.data.get_sensor("left_waste_F"))

        ####reward####
        reward = alive_bonus

        reward -= energy_coefficient*abs(energy_a - energy_b)

        reward += forward_coefficient*self.sim.data.qvel[0]

        joint = abs((joint_after - joint_before))/self.dt 
        reward += min(mini,joint_coefficient*joint)

        if(posafter < posbefore):
            reward -= penalty
            pena = -penalty

        if sym_flag:
            reward -= symmetry_coefficient*symmetry 

        # encourage more efficient learning
        if num<500:
            reward -= abs(ang)
            if(self.sim.data.qpos[0]<0.5):
                reward -= 0.5*alive_bonus
            if(abs(joint_after - joint_before)/self.dt<0.05):
                reward -= 1.0
        
        waru = max(1,1+forward_coefficient)
        reward = reward/waru

        rdm = random.randint(1, 1000)
        if rdm == 100 and view:
            print('=======================================================')
            print('reward = {:.3f}    num={}'.format(reward,num))
            print('-------------------------------------------------------')
            #print('force       : -{:.4f}'.format(force_coefficient*force))
            print('alive_bonus : +{:.1f}'.format(alive_bonus))
            print('forward_pena: {:.1f}'.format(pena))
            print('joint       : +{:.2f}'.format(min(mini,joint_coefficient*joint)))
            print('velocity    : +{:.2f}'.format(forward_coefficient*self.sim.data.qvel[0]))
            print('energy      : -{:.2f}'.format(energy_coefficient*abs(energy_a-energy_b)))
            if sym_flag:
                print('symme       : -{:.2f}'.format(symmetry_coefficient*symmetry))
            print('-------------------------------------------------------')
            print("energy = |{:.2f} - {:.2f}| = {:.2f}   w1={}".format(energy_a,energy_b, abs(energy_a-energy_b), energy_coefficient))
            print('velocity = {:.2f}   w2={}'.format(self.sim.data.qvel[0],forward_coefficient))
            print("joint = {:.2f}   w3={}".format(abs((joint_after - joint_before)) / self.dt, joint_coefficient))
            print('a_before = {}'.format(a_before))
            print('    a    = {}'.format(a))
            if sym_flag:
                print('a_simmentry = {}   w5={}'.format(a_compare,symmetry_coefficient))
            print('pos_x = {:.3f}'.format(posafter))
            print('=======================================================')
            print('')

        # done = False
        done = not (ang > -1.4 and ang < 1.4)
        ob = self._get_obs(w2)
        return ob, reward, done, {}

    def sensor(self, flag, writer,header=False):
        if flag:
            gear = self.geer
            right_contact = 0
            left_contact = 0
            right_x_N = 0
            left_x_N = 0
            right_z_N = 0
            left_z_N = 0

            # energy
            energy,kinematic,potential,elastic = self.energy_measure()

            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
                if(int(contact.geom2)<30):
                    right_x_N += -c_array[2]
                    right_z_N += c_array[0]
                    right_contact = 1
                if(int(contact.geom2)>30):
                    left_x_N += -c_array[2]
                    left_z_N += c_array[0]
                    left_contact = 1


            data = [gear*self.sim.data.get_sensor('waste_F'),
                    gear*self.sim.data.get_sensor('left_waste_F'),
                    self.sim.data.qpos[0],
                    self.sim.data.qvel[0],
                    self.sim.data.qpos[1],
                    kinematic,
                    elastic,
                    potential,
                    energy,
                    right_contact,
                    left_contact,
                    right_x_N,
                    left_x_N,
                    right_z_N,
                    left_z_N,
                    self.sim.data.get_site_xpos('s_hip')[0],
                    self.sim.data.get_site_xpos('s_hip')[2],
                    self.sim.data.get_site_xpos('s_waste')[0],
                    self.sim.data.get_site_xpos('s_waste')[2],
                    self.sim.data.get_site_xpos('s_left_waste')[0],
                    self.sim.data.get_site_xpos('s_left_waste')[2],
                    self.sim.data.get_site_xpos('s_thigh')[0],
                    self.sim.data.get_site_xpos('s_thigh')[2],
                    self.sim.data.get_site_xpos('s_left_thigh')[0],
                    self.sim.data.get_site_xpos('s_left_thigh')[2],
                    self.sim.data.get_joint_qpos("waste_joint"),
                    self.sim.data.get_joint_qpos("left_waste_joint"),
                    self.sim.data.get_joint_qpos("rooty"),
                    ]

            if header:
                data = ['waste_F',
                        'left_waste_F',
                        'x_pos',
                        'x_vel',
                        'z_pos',
                        'kinematic',
                        'elastic',
                        'potential',
                        'energy',
                        'right_contact',
                        'left_contact',
                        'right_x_N',
                        'left_x_N',
                        'right_z_N',
                        'left_z_N',
                        's_hip_x',
                        's_hip_z',
                        's_waste_x',
                        's_waste_z',
                        's_left_waste_x',
                        's_left_waste_z',
                        's_thigh_x',
                        's_thigh_z',
                        's_left_thigh_x',
                        's_left_thigh_z',
                        'waste_joint_angle',
                        'left_waste_joint_angle',
                        'rooty'
                        ]
            writer.writerow(data)

        if flag==False:
            writer.writerow('')

    def _get_obs(self,w2):
        obs = np.array([
            np.clip(self.sim.data.get_joint_qvel("rootx"), -10, 10),
            self.sim.data.get_joint_qpos("rooty"),
            np.clip(self.sim.data.get_joint_qvel("rooty"), -10, 10),
            self.sim.data.get_joint_qpos("waste_joint"),
            self.sim.data.get_joint_qpos("left_waste_joint"),
            np.clip(self.sim.data.get_joint_qvel("waste_joint"), -10, 10),
            np.clip(self.sim.data.get_joint_qvel("left_waste_joint"), -10, 10),
            self.sim.data.get_joint_qpos("waste_z"),
            self.sim.data.get_joint_qpos("left_waste_z"),
            np.clip(self.sim.data.get_joint_qvel("waste_z"), -10, 10),
            np.clip(self.sim.data.get_joint_qvel("left_waste_z"), -10, 10),
            np.clip(self.sim.data.get_joint_qvel("foot_slide"), -10, 10),
            np.clip(self.sim.data.get_joint_qvel("left_foot_slide"), -10, 10),
            ])
        obs = np.hstack([obs, w2])
        return obs

    def reset_model(self,num):
        if(num<1000):
            self.sim.data.set_joint_qpos("waste_joint", -0.33)
            self.sim.data.set_joint_qpos("left_waste_joint", 0.279)
            self.sim.data.set_joint_qpos("rootz", -0.01)
            self.sim.data.set_joint_qpos("waste_z", -0.11)
            self.sim.data.set_joint_qvel("rootz", -3)
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            qpos = qpos + \
                self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
            qvel = qvel + \
                self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            self.set_state(qpos, qvel)
        elif(num<1500):
            rdm = random.randint(1,3)
            if(rdm == 1):
                self.sim.data.set_joint_qpos("waste_joint", -0.33)
                self.sim.data.set_joint_qpos("left_waste_joint", 0.28)
                self.sim.data.set_joint_qpos("rootz", -0.01)
                self.sim.data.set_joint_qpos("waste_z", -0.11)
                self.sim.data.set_joint_qvel("rootz", random.uniform(-3,0))
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                qpos = qpos + \
                    self.np_random.uniform(low=-.05, high=.05, size=self.model.nq)
                qvel = qvel + \
                    self.np_random.uniform(low=-.05, high=.05, size=self.model.nv)
                self.set_state(qpos, qvel)
            else:
                self.sim.data.set_joint_qpos("waste_joint", 0.28)
                self.sim.data.set_joint_qpos("left_waste_joint", -0.33)
                self.sim.data.set_joint_qpos("rootz", -0.01)
                self.sim.data.set_joint_qpos("left_waste_z", -0.11)
                self.sim.data.set_joint_qvel("rootz", random.uniform(-3,0))
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                qpos = qpos + \
                    self.np_random.uniform(low=-.05, high=.05, size=self.model.nq)
                qvel = qvel + \
                    self.np_random.uniform(low=-.05, high=.05, size=self.model.nv)
                self.set_state(qpos, qvel)
        elif num<4000:
            rdm = random.randint(1,2)
            if(rdm == 1):
                self.sim.data.set_joint_qpos("rooty", random.uniform(-0.05,0.4))
                self.sim.data.set_joint_qpos("waste_joint", random.uniform(0.15,0.6))
                self.sim.data.set_joint_qpos("left_waste_joint", random.uniform(-0.4,-0.1))
                self.sim.data.set_joint_qpos("rootz", -0.01)
                self.sim.data.set_joint_qpos("waste_z", random.uniform(-0.11,0.11))
                self.sim.data.set_joint_qvel("rootz", random.uniform(-3,0))
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                self.set_state(qpos, qvel)
            else:
                self.sim.data.set_joint_qpos("rooty", random.uniform(-0.05,0.4))
                self.sim.data.set_joint_qpos("waste_joint",random.uniform(-0.4,-0.1))
                self.sim.data.set_joint_qpos("left_waste_joint", random.uniform(0.15,0.6))
                self.sim.data.set_joint_qpos("rootz", -0.01)
                self.sim.data.set_joint_qpos("left_waste_z", random.uniform(-0.11,0.11))
                self.sim.data.set_joint_qvel("rootz", random.uniform(-3,0))
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                self.set_state(qpos, qvel)
        else:
            rdm = random.randint(1,2)
            if(rdm == 1):
                self.sim.data.set_joint_qpos("rooty", random.uniform(-0.05,0.4))
                self.sim.data.set_joint_qpos("waste_joint", random.uniform(0.15,0.6))
                self.sim.data.set_joint_qpos("left_waste_joint", random.uniform(-0.4,-0.1))
                self.sim.data.set_joint_qpos("rootz", -0.01)
                self.sim.data.set_joint_qpos("waste_z", random.uniform(-0.11,0.11))
                self.sim.data.set_joint_qvel("rootz", random.uniform(-3,0))
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                self.set_state(qpos, qvel)
            else:
                self.sim.data.set_joint_qpos("rooty", random.uniform(-0.05,0.4))
                self.sim.data.set_joint_qpos("waste_joint",random.uniform(-0.4,-0.1))
                self.sim.data.set_joint_qpos("left_waste_joint", random.uniform(0.15,0.6))
                self.sim.data.set_joint_qpos("rootz", -0.01)
                self.sim.data.set_joint_qpos("left_waste_z", random.uniform(-0.11,0.11))
                self.sim.data.set_joint_qvel("rootz", random.uniform(-3,0))
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                self.set_state(qpos, qvel)
        return self._get_obs(0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def pos(self):
        return self.sim.data.qpos[0]
        
    def vel(self):
        return self.sim.data.get_joint_qvel("rootx")

    def touch_check(self):
        right_touch = False
        left_touch = False

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            if(int(contact.geom2)<30):
                right_touch = True
            if(int(contact.geom2)>30):
                left_touch = True
        return right_touch, left_touch
