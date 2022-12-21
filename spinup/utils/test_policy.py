import time
import joblib
import os
import os.path as osp
import torch
from spinup.utils.logx import EpochLogger
import numpy as np
import PySimpleGUI as sg  # utilized for stting GUI environment 
import csv

def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action

def take_data(env,flag=False):
    if flag == False:
        csvfile = None
        writer = None
    else:
        a, i = 1, 1
        while a<100:
            a += 1
            if(str(a) in str(env)):
                file_path = os.join(os.getcwd(),'gait_data/trial{}'.format(a))

        if os.path.exists(file_path) is False:
            os.mkdir(file_path)
        
        while True:
            if(os.path.exists(file_path + '/data{}.csv'.format(i)) == False):
                make_file = file_path + '/data{}.csv'.format(i)
                # pathlib.Path(make_file).touch()
                with open(make_file,"w"):pass
                break
            else:
                i = i+1
        csvfile = open(make_file,'w',encoding='utf-8')
        writer = csv.writer(csvfile)
    return csvfile, writer



def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    # GUI
    sg.theme('BlueMono')
    layout = [
    [sg.Slider(
        range=(0.0, 2.50),
        default_value = 0,
        resolution = 0.05,
        orientation = 'h',
        size = (60,30),
        font=('Arial',10),
        enable_events=True,
        key = 'slider1')],
    [sg.Text('$\omega_{v}$',size=(0,1),key='OUTPUT3',font=('Arial',30))],
    [sg.Text('Instant Velocity',size=(0,1),key='OUTPUT2',font=('Arial',30))],
    [sg.Text('Average Velocity',size=(0,1),key='OUTPUT',font=('Arial',30))],
        ]
    window = sg.Window('window title',layout)

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    a_before = np.zeros(2)
    p = 0 
    alive_flag = True
    csvfile, writer = take_data(env, flag=True)
    if csvfile is not None:
        env.sensor(alive_flag,writer,header=True)
    vel_flag = False

    while n < num_episodes:
        # detect input from gui
        event, val = window.read(timeout=0.1)
        if vel_flag:
            window['OUTPUT'].update(value=str(format(mean_vel,'.2f'))+'[m/s] (average)')
        vel = env.vel()
        window['OUTPUT2'].update(value=str(format(vel,'.2f'))+'[m/s]')
        window['OUTPUT3'].update('omega_v = '+str(format(p,'.2f')))
        window['slider1'].update(float(format(p,'.2f')))
        if event == sg.WIN_CLOSED:
            break
        if event is None:
            break
        if event == 'slider1':
            p = (val['slider1'])

        # render = False
        if render:
            env.render()
            window.BringToFront
            time.sleep(1e-3)

        o[-1] = p
        a = get_action(o)
        o, r, d, _ = env.step(a,a_before=a_before,w2=p,view=False)
        a_before = a

        start = 0
        if ep_len == 0:
            ep_measure = 0
        else:
            ep_measure += 1
            distance = env.pos()-start
            mean_vel = distance/0.008/ep_measure
            vel_flag = True

        ep_ret += r
        ep_len += 1
        alive_flag = True

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(num=1000), 0, False, 0, 0
            mean_vel = 0

        # taking data
        if csvfile is not None:
            env.sensor(alive_flag,writer)

    window.close()
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    csvfile.close()


