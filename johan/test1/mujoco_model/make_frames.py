import os
import math
import mujoco
import signal
import mujoco_viewer
import numpy as np


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


class MotionElem:

    def __init__(self, t_vals, azimuth_vals, elevation_vals, distance_vals):
        self.t_vals = t_vals
        self.azimuth_vals = azimuth_vals
        self.elevation_vals = elevation_vals
        self.distance_vals = distance_vals

    def azimuth(self, t):
        value = np.interp(t, self.t_vals, self.azimuth_vals, left=self.azimuth_vals[0], right=self.azimuth_vals[1])
        return value

    def elevation(self, t):
        value = np.interp(t, self.t_vals, self.elevation_vals, left=self.elevation_vals[0], right=self.elevation_vals[1])
        return value

    def distance(self,t):
        value = np.interp(t, self.t_vals, self.distance_vals, left=self.distance_vals[0], right=self.distance_vals[1])
        return value

    def active(self,t):
        if t >= self.t_vals[0] and t < self.t_vals[1]:
            return True
        else:
            return False



# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    model = mujoco.MjModel.from_xml_path('model.xml')
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    #viewer.render_mode = 'window'
    viewer.render_mode = 'offscreen'

    signal.signal(signal.SIGINT, sigint_handler)
    
    viewer.cam.distance = 50 
    viewer.cam.azimuth =  180 
    viewer.cam.elevation = -4
    viewer.cam.lookat = [0.0, 0.0, 0.0]

    period = 0.5
    sla_amplitude = np.deg2rad(5.0)
    sla_phase = np.deg2rad(180.0)
    sla_offset = sla_amplitude 
    

    motion_program = [
            MotionElem( ( 0*period,  2*period), (180.0, 180.0), (-4.0,   -4.0), (50, 50) ),
            MotionElem( ( 3*period,  6*period), (180.0, 180.0), (-4.0,  -21.0), (50, 50) ),
            MotionElem( ( 6*period,  9*period), (180.0, 270.0), (-21.0, -21.0), (50, 50) ),
            MotionElem( ( 9*period, 12*period), (270.0, 360.0), (-21.0, -21.0), (50, 30) ),
            MotionElem( (12*period, 15*period), (360.0, 360.0), (-21.0, -21.0), (30, 30) ),
            MotionElem( (15*period, 18*period), (360.0, 270.0), (-21.0, -21.0), (30, 30) ),
            MotionElem( (18*period, 21*period), (270.0, 180.0), (-21.0, -5.0),  (30, 50) ),
            ]

    frame_count = 0
    output_dir = 'frames'
    os.makedirs(os.path.join(os.curdir, output_dir), exist_ok=True)

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
            
        if frame is not None:
            filename = os.path.join(output_dir, f'frame_{frame_count:06d}.npy')
            print(filename)
            np.save(filename, frame)
            frame_count += 1

        for elem in motion_program:
            if elem.active(data.time):
                viewer.cam.azimuth = elem.azimuth(data.time)
                viewer.cam.elevation = elem.elevation(data.time)
                viewer.cam.distance = elem.distance(data.time)
        if data.time > elem.t_vals[1]:
            done = True

    viewer.close()


