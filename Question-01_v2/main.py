import numpy as np
import astropy.units as u
from einsteinpy.plotting import StaticGeodesicPlotter
from einsteinpy.coordinates import SphericalDifferential
from einsteinpy.bodies import Body
from einsteinpy.geodesic import Geodesic
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

Attractor = Body(mass=100 * u.M_sun, parent=None)

distance = 69.817e6 * u.m # distance of mercury at aphelion = r
speed_at_aphelion = 38.86e6 * u.m / u.s # speed = v
omega = (u.rad * speed_at_aphelion) / distance # owega = v / r

if os.path.isfile('sph_obj.bin'):
    with open('sph_obj.bin', 'rb') as h:
        sph_obj = pickle.load(h)
    print('loaded sph_obj')
else:
    with open('sph_obj.bin', 'wb') as f:
        sph_obj = SphericalDifferential(r=distance, theta=np.pi/2*u.rad, phi=-np.pi/8*u.rad,
                                        v_r=0.*u.m/u.s, v_t=0.*u.rad/u.s, v_p=omega)
        pickle.dump(sph_obj, f)

if os.path.isfile('Object.bin'):
    with open('Object.bin', 'rb') as h:
        Object = pickle.load(h)
    print('loaded Object')
else:
    with open('Object.bin', 'wb') as f:
        Object = Body(differential=sph_obj, parent=Attractor)
        pickle.dump(Object, f)

t = 0 * u.s
start_lambda = 0.0
end_lambda = 0.2
step_size = 5e-8

if os.path.isfile('geo.bin'):
    with open('geo.bin', 'rb') as h:
        geo = pickle.load(h)
    print('loaded geo')
else:
    with open('geo.bin', 'wb') as f:
        geo = Geodesic(Object, time=t, end_lambda=end_lambda, step_size=step_size)
        pickle.dump(geo, f)

if os.path.isfile('trajectory.csv'):
    print('loaded trajectory')
    trajectory = np.loadtxt('trajectory.csv', delimiter=',')
else:
    trajectory = geo.trajectory
    np.savetxt('trajectory.csv', trajectory, delimiter=',')

data_x = np.array([coord[1] for coord in trajectory]).reshape(-1, 1)
x = MinMaxScaler(feature_range=(-1,1))
x.fit(data_x)
data_x = np.array([i for i in x.transform(data_x).ravel()])

data_y = np.array([coord[2] for coord in trajectory]).reshape(-1, 1)
y = MinMaxScaler(feature_range=(-1,1))
y.fit(data_y)
data_y = np.array([i for i in y.transform(data_y).ravel()])

data_z = np.array([coord[3] for coord in trajectory]).reshape(-1, 1)
z = MinMaxScaler(feature_range=(-1,1))
z.fit(data_z)
data_z = np.array([i for i in z.transform(data_z).ravel()])

fig, ax = plt.subplots(figsize=(6, 6))
ax = plt.axes(projection="3d")

attractor_radius_scale=-1.0
attractor_color="#ffcc00"
use_3d=True
ani = None

ax.set_zlabel("$z$ (m)")
ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
attractor_present = False

def _mindist(x, y, z=0):
    return np.sqrt(x * x + y * y + z * z)

def _draw_attractor(radius, xarr, yarr):
    attractor_present = True
    if use_3d:
        ax.plot([0], [0], [0], "o", mew=0, color=attractor_color)
    elif attractor_radius_scale == -1.0:
        minrad_nooverlap = _mindist(xarr[0], yarr[0])
        for i, _ in enumerate(xarr):
            minrad_nooverlap = min(
                minrad_nooverlap, _mindist(xarr[i], yarr[i])
            )

        xlen = max(xarr) - min(xarr)
        ylen = max(yarr) - min(yarr)
        minlen_plot = min(xlen, ylen)
        min_radius = minlen_plot / 12

        radius = min(min_radius, minrad_nooverlap)
        get_curr_plot_radius = radius
        ax.add_patch(Circle((0, 0), radius, lw=0, color=attractor_color))
    else:
        radius = radius * attractor_radius_scale
        get_curr_plot_radius = radius
        ax.add_patch(
            Circle((0, 0), radius.value, lw=0, color=attractor_color)
        )

def _set_scaling(x_range, y_range, z_range, lim):
    if x_range < lim and y_range < lim and z_range < lim:
        return
    if x_range < lim:
        ax.set_xlim([-lim, lim])
    if y_range < lim:
        ax.set_ylim([-lim, lim])
    if z_range < lim:
        ax.set_zlim([-lim, lim])

def plot(x, y, z, geodesic, color="#{:06x}".format(random.randint(0, 0xFFFFFF)), save=False):
    if not use_3d:
        ax.plot(x, y, "--", color=color)
        ax.plot(x[-1], y[-1], "o", mew=0, color=color)
    else:
        x_range = max(x) - min(x)
        y_range = max(y) - min(y)
        z_range = max(z) - min(z)
        _set_scaling(x_range, y_range, z_range, 1e-5)
        ax.plot(x, y, z, "--", color=color)
        ax.plot(x[-1:], y[-1:], z[-1:], "o", mew=0, color=color)

    if not attractor_present:
        _draw_attractor(geodesic.metric.scr, x, y)

    if save:
        plt.savefig('path.png')

def animate(
    x, y, z, geodesic, color="#{:06x}".format(random.randint(0, 0xFFFFFF)), interval=50
):
    x_max, x_min = max(x), min(x)
    y_max, y_min = max(y), min(y)
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.2
    frames = x.shape[0]
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)

    if use_3d:
        z_max, z_min = max(z), min(z)
        margin_z = (z_max - z_min) * 0.1     
        ax.set_zlim(z_min - margin_z, z_max + margin_z)   

    if not use_3d:
        (pic,) = ax.plot([], [], "--", color=color)
    else:
        (pic,) = ax.plot([], [], [], "--", color=color)

    if not attractor_present:
        _draw_attractor(geodesic.metric.scr, x, y)

    if use_3d:
        def _update(frame):
            pic.set_data_3d(x[: frame + 1],y[: frame + 1],z[: frame + 1])
            return (pic,)
    else:
        def _update(frame):
            pic.set_xdata(x[: frame + 1])
            pic.set_ydata(y[: frame + 1])
            return (pic,)
    global ani
    ani = FuncAnimation(
        fig, _update, frames=frames, interval=interval, blit=True, save_count=0
    )


plot(data_x, data_y, data_z, geo, save=True)
# animate(data_x, data_y, data_z, geo, interval=0.01)
# ani.save('animation.mp4')
plt.show()