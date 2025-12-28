import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, RadioButtons

# --- 기초 물리 함수 ---
def normalize(v): 
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def reflect(i, n): 
    return i - 2 * np.dot(i, n) * n

def refract(i, n, n1, n2):
    eta = n1 / n2
    cos_i = -np.dot(n, i)
    sin2_t = eta**2 * (1 - cos_i**2)
    if sin2_t > 1.0: return reflect(i, n) # 전반사
    cos_t = np.sqrt(1.0 - sin2_t)
    return eta * i + (eta * cos_i - cos_t) * n

def intersect_tri(o, d, v0, v1, v2):
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d, e2)
    a = np.dot(e1, h)
    if -1e-8 < a < 1e-8: return None
    f = 1.0/a
    s = o - v0
    u = f * np.dot(s, h)
    if u < 0 or u > 1: return None
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0 or u + v > 1: return None
    t = f * np.dot(e2, q)
    return t if t > 0.1 else None

class Box3D:
    def __init__(self, center, size, color, name, is_room=False, is_glass=False):
        self.center = np.array(center, dtype=float)
        self.size = np.array(size)
        self.color = np.array(color)
        self.name = name
        self.is_room = is_room
        self.is_glass = is_glass
        self.ior = 1.45
        self.update()

    def update(self):
        s = self.size / 2
        v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * s
        self.v = v + self.center
        self.tris = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        self.faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]

    def intersect(self, o, d):
        min_t, hit_n = float('inf'), None
        for f in self.tris:
            t = intersect_tri(o, d, self.v[f[0]], self.v[f[1]], self.v[f[2]])
            if t and t < min_t:
                min_t = t
                n = normalize(np.cross(self.v[f[1]]-self.v[f[0]], self.v[f[2]]-self.v[f[0]]))
                hit_n = n if np.dot(d, n) < 0 else -n
        return min_t, hit_n

class RayTracer3D:
    def __init__(self):
        self.cam_pos = np.array([0.0, -8.0, 7.0])
        self.light_z, self.light_radius, self.light_power = 9.8, 6.0, 2500.0
        self.ambient = np.array([0.05, 0.05, 0.07])
        self.objects = [
            Box3D([0, 0, 0], [20, 20, 20], [0.95, 0.95, 1.0], "Room", is_room=True),
            Box3D([-4, 2, -3], [5, 5, 5], [0.0, 0.5, 1.0], "GlassBox", is_glass=True),
            Box3D([4, 2, -3], [5, 5, 5], [1.0, 0.1, 0.1], "SolidBox")
        ]
        self.target_obj = self.objects[1]
        self.res = 50
        self.ray_logs = {}
        self.selected_pixel = None
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 9))
        self.ax3 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122)
        self.fig.subplots_adjust(bottom=0.2)
        
        # UI Setup
        self.btn_ax_render = self.fig.add_axes([0.4, 0.05, 0.1, 0.05])
        self.btn_render = Button(self.btn_ax_render, 'RENDER', color='#2ecc71')
        self.btn_render.on_clicked(self.run_render)
        
        self.btn_ax_check = self.fig.add_axes([0.52, 0.05, 0.1, 0.05])
        self.btn_check = Button(self.btn_ax_check, 'CHECK', color='#3498db')
        self.btn_check.on_clicked(self.analyze_ray)
        
        self.radio_ax = self.fig.add_axes([0.05, 0.05, 0.15, 0.08])
        self.radio = RadioButtons(self.radio_ax, ('GlassBox', 'SolidBox'))
        self.radio.on_clicked(self.select_obj)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_view()

    def select_obj(self, label):
        self.target_obj = self.objects[1] if label == 'GlassBox' else self.objects[2]
        self.update_view()

    def trace(self, o, d, depth, dist_t=0, n1=1.0, log=None):
        if depth > 5 or np.any(np.abs(o) > 25): return [], self.ambient
        
        # 광원 충돌
        if d[2] > 0:
            t_l = (self.light_z - o[2]) / d[2]
            if 0 < t_l < 30:
                p_l = o + d * t_l
                if np.linalg.norm(p_l[:2]) < self.light_radius:
                    att = self.light_power / ((dist_t + t_l)**2 + 150)
                    col = np.array([1, 0.98, 0.9]) * att + self.ambient
                    if log is not None: log.append({'p': p_l, 'type': 'Light', 'val': col.mean()})
                    return [p_l], col

        closest_t, hit_n, hit_obj = float('inf'), None, None
        for obj in self.objects:
            t, n = obj.intersect(o, d)
            if t < closest_t: closest_t, hit_n, hit_obj = t, n, obj
            
        if hit_obj and closest_t < 40:
            hp = o + d * closest_t
            if hit_obj.is_glass:
                n2 = hit_obj.ior if n1 == 1.0 else 1.0
                if log is not None: log.append({'p': hp, 'type': 'Refract', 'val': n2})
                path, col = self.trace(hp, refract(d, hit_n, n1, n2), depth+1, dist_t+closest_t, n2, log)
                absorb = np.exp(-0.02 * closest_t) # 투명도 개선
                final_col = (col * 0.92 + hit_obj.color * 0.08) * absorb
                return [hp]+path, np.clip(final_col, 0, 1)
            else:
                refl_coeff = 0.6 if not hit_obj.is_room else 0.1
                if log is not None: log.append({'p': hp, 'type': 'Reflect', 'val': refl_coeff})
                path, col = self.trace(hp, reflect(d, hit_n), depth+1, dist_t+closest_t, 1.0, log)
                final_col = col * refl_coeff + hit_obj.color * 0.1
                return [hp]+path, np.clip(final_col, 0, 1)
                
        return [o + d * 5], self.ambient

    def run_render(self, event=None):
        self.selected_pixel = None
        self.ax2.clear()
        self.ray_logs = {}
        img = np.zeros((self.res, self.res, 3))
        grid = np.linspace(-7.5, 7.5, self.res)
        for i, y in enumerate(reversed(grid)):
            for j, x in enumerate(grid):
                rd = normalize(np.array([x, 8, y-5]))
                l = []
                path, col = self.trace(self.cam_pos, rd, 0, log=l)
                img[i, j] = col
                self.ray_logs[(i, j)] = (path, l, col)
        self.ax2.imshow(np.clip(img * 2.5, 0, 1), interpolation='bilinear')
        self.update_view()

    def on_click(self, event):
        if event.inaxes == self.ax2:
            self.selected_pixel = (int(event.ydata + 0.5), int(event.xdata + 0.5))

    def analyze_ray(self, event=None):
        if self.selected_pixel in self.ray_logs:
            self.update_view(analyze=True)

    def update_view(self, analyze=False):
        self.ax3.clear()
        self.ax3.set_box_aspect([1,1,1])
        for obj in self.objects:
            f_list = [obj.v[list(f)] for f in obj.faces]
            a = 0.05 if obj.is_glass else (0.01 if obj.is_room else 0.2)
            self.ax3.add_collection3d(Poly3DCollection(f_list, alpha=a, facecolors=obj.color, edgecolors='w', lw=0.1))
        
        if analyze and self.selected_pixel:
            path, logs, col = self.ray_logs[self.selected_pixel]
            full_p = [self.cam_pos] + path
            self.ax3.plot([p[0] for p in full_p], [p[1] for p in full_p], [p[2] for p in full_p], color='yellow', lw=3, zorder=15)
            for l in logs:
                self.ax3.text(l['p'][0], l['p'][1], l['p'][2], f"{l['type']}\n({l['val']:.2f})", color='cyan', fontsize=8, fontweight='bold')
        elif hasattr(self, 'ray_logs'):
            for k, v in self.ray_logs.items():
                if k[0]%6==0 and k[1]%6==0:
                    p_data, _, col = v
                    full_p = [self.cam_pos] + p_data
                    brightness = np.max(col)
                    if brightness > 0.05: # 반사광 가시성 문턱값 하향
                        self.ax3.plot([x[0] for x in full_p], [x[1] for x in full_p], [x[2] for x in full_p], 
                                     color=np.clip(col*1.8, 0, 1), alpha=np.clip(brightness*1.5, 0.2, 0.8), lw=0.8)
        
        self.ax3.set_xlim(-10, 10); self.ax3.set_ylim(-10, 10); self.ax3.set_zlim(-10, 10)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        s = 0.5
        if event.key == 'up': self.target_obj.center[1]+=s
        elif event.key == 'down': self.target_obj.center[1]-=s
        elif event.key == 'left': self.target_obj.center[0]-=s
        elif event.key == 'right': self.target_obj.center[0]+=s
        elif event.key == 'w': self.target_obj.center[2]+=s
        elif event.key == 's': self.target_obj.center[2]-=s
        self.target_obj.update(); self.update_view()

rt = RayTracer3D()
plt.show()