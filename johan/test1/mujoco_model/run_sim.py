import os
import time
import math
import mujoco
import signal
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt


done = False

def sigint_handler(signum, frame):
    """
    SIGINT handler. Sets done to True to quit simulation.
    """
    global done
    done = True


def soft_start(t,ts):
    """
    Soft startup function for actuators. Ramps from 0.0 to 1.0 during interval from t=0
    to t=ts.
    """
    rval = 0.0
    if t < ts:
        rval = t/ts
    else:
        rval = 1.0
    return rval


def sin_motion(t, amplitude, phase, offset, period):
    start_value  = soft_start(data.time, period)
    pos = amplitude*math.sin(2.0*math.pi*t/period + phase) + offset
    vel = (2.0*math.pi/period)*amplitude*math.cos(2.0*math.pi*t/period + phase)
    pos *= start_value
    vel *= start_value
    return pos, vel

def cos_motion(t, amplitude, phase, offset, period):
    start_value  = soft_start(data.time, period)
    pos = amplitude*math.cos(2.0*math.pi*t/period + phase) + offset
    vel = -(2.0*math.pi/period)*amplitude*math.sin(2.0*math.pi*t/period + phase)
    pos *= start_value
    vel *= start_value
    return pos, vel


# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    model = mujoco.MjModel.from_xml_path('model.xml')
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    viewer.render_mode = 'window'
    #viewer.render_mode = 'offscreen'

    print(viewer)

    signal.signal(signal.SIGINT, sigint_handler)
    
    viewer.cam.distance = 50 
    viewer.cam.azimuth =  20 
    viewer.cam.elevation = -45 
    viewer.cam.lookat = [0.0, 0.0, 0.0]
    
    period = 0.5
    sla_amplitude = np.deg2rad(5.0)
    sla_phase = np.deg2rad(180.0)
    sla_offset = sla_amplitude 


    while not done:

        mujoco.mj_step(model, data)
        
        sla_setpos, sla_setvel = cos_motion(
                data.time, 
                sla_amplitude, 
                sla_phase, 
                sla_offset, 
                period
                )

        data.actuator('sla_position').ctrl = sla_setpos
        data.actuator('sla_velocity').ctrl = sla_setvel

        try:
            if viewer.render_mode == 'window':
                frame = None
                viewer.render()
            else:
                frame = viewer.read_pixels()
        except:
            done = True

        #time.sleep(0.01)
    viewer.close()


