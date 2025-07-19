#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPC-based obstacle avoidance + Pure-Pursuit tracker
 ├ 장애물 : /cluster_markers   (Marker)
 ├ GPS   : /gps_path           (Path)
 ├ α각   : /alpha_angle_topic  (Float32)
 └→ 최종 조향 /ldr_stragl (Float32, −1‥+1)

주요 특성
─────────
• 장애물 토픽이 끊겨도 50 ms 타이머에서 계속 주행·시각화
• enable_viz 파라미터로 OpenCV 창을 언제든 ON/OFF
"""

import math, time, cv2, rclpy, numpy as np
from rclpy.node             import Node
from std_msgs.msg           import Float32
from visualization_msgs.msg import Marker
from nav_msgs.msg           import Path
from scipy.ndimage          import gaussian_filter1d

class MPCPPNode(Node):
    # ── 상수(잘 안 바뀌는 값) ────────────────────────────
    BIN_EDGES   = np.arange(-80, 82, 2)
    GAUSS_SIG   = 4.5
    IMG_SZ, SCALE_PX = 800, 75
    CXY         = (IMG_SZ // 2, IMG_SZ // 2)
    MARKER_TOUT = 0.2               # [s] 장애물 메시지 타임아웃
    TIMER_HZ    = 20.0              # [Hz] 메인 루프

    def __init__(self):
        super().__init__('mpc_pp_node')

        # ── 런타임 파라미터 선언 ────────────────────────
        dp = self.declare_parameter
        dp('enable_viz',          True)
        dp('gps_yaw_offset_deg', -90.0)
        dp('gps_weight',          0.05)
        dp('obs_weight',          5.0)
        dp('smooth_weight',       0.01)
        dp('base_radius',         0.6)
        dp('wheelbase',           1.0)
        dp('lookahead_pp',        2.0)
        dp('norm_deg',           18.0)
        dp('steer_sign',         -1.0)
        dp('max_steer_deg',      18.0)
        dp('dt',                  0.1)
        dp('horizon',            23)
        dp('sim_speed',           1.5)

        # ── 파라미터 값 읽기 ───────────────────────────
        gp = self.get_parameter
        self.viz        = gp('enable_viz').value
        self.yaw_off    = math.radians(gp('gps_yaw_offset_deg').value)
        self.w_gps      = gp('gps_weight').value
        self.w_obs      = gp('obs_weight').value
        self.w_smooth   = gp('smooth_weight').value
        self.base_r     = gp('base_radius').value
        self.wheelbase  = gp('wheelbase').value
        self.Ld_pp      = gp('lookahead_pp').value
        self.norm_deg   = gp('norm_deg').value
        self.sign       = gp('steer_sign').value
        self.max_deg    = gp('max_steer_deg').value
        self.dt_mpc     = gp('dt').value
        self.N_mpc      = int(gp('horizon').value)
        self.speed_mpc  = gp('sim_speed').value

        self.get_logger().info(
            f"node up • viz:{self.viz}  w_obs:{self.w_obs}  w_gps:{self.w_gps}"
        )

        # ── ROS 통신 설정 ──────────────────────────────
        self.create_subscription(Marker , '/cluster_markers',   self.cb_marker, 10)
        self.create_subscription(Path   , '/gps_path',          self.cb_path,   10)
        self.create_subscription(Float32, '/alpha_angle_topic', self.cb_alpha,  10)
        
        # Publisher
        self.steer_pub = self.create_publisher(Float32, '/ldr_stragl', 10)
        #self.steer_pub = self.create_publisher(Float32, '/steering_angle', 10)



        # 파라미터 동적 변경 감지 – enable_viz 전용
        self.add_on_set_parameters_callback(self.on_set_param)

        # ── 상태 변수 ──────────────────────────────────
        self.alpha_deg       = 0.0
        self.prev_opt_deg    = 0.0
        self.gps_pts         = []          # [(x,y)…]
        self.obstacles       = []          # [(x,y,r)…]
        self.histogram       = np.zeros(len(self.BIN_EDGES)-1)
        self.last_marker_t   = 0.0

        if self.viz:
            cv2.namedWindow("Predicted Path", cv2.WINDOW_NORMAL)

        # 주기 루프
        self.create_timer(1.0 / self.TIMER_HZ, self.main_loop)

    # ── 동적 파라미터 변경(특히 enable_viz) ─────────────
    def on_set_param(self, params):
        for p in params:
            if p.name == 'enable_viz':
                if bool(p.value) and not self.viz:
                    cv2.namedWindow("Predicted Path", cv2.WINDOW_NORMAL)
                    self.get_logger().info("Visualization ENABLED")
                if not bool(p.value) and self.viz:
                    cv2.destroyWindow("Predicted Path")
                    self.get_logger().info("Visualization DISABLED")
                self.viz = bool(p.value)
        return rclpy.parameter.SetParametersResult(successful=True)

    # ╭─── 수신 콜백 : 데이터만 저장 ─────────────────────╮
    def cb_alpha(self, msg):  
        self.alpha_deg = msg.data

    def cb_path(self, msg: Path):
        arr = np.array([[-p.pose.position.y, p.pose.position.x] for p in msg.poses],
                       dtype=float)
        if arr.size and self.yaw_off:
            c, s = math.cos(self.yaw_off), math.sin(self.yaw_off)
            arr  = (np.array([[c, -s], [s,  c]]) @ arr.T).T
        self.gps_pts = arr.tolist()

    def cb_marker(self, mk: Marker):
        self.last_marker_t = time.time()
        if not mk.points:
            self.obstacles = []; self.histogram[:] = 0
            return

        h = np.zeros(len(self.BIN_EDGES)-1); obs=[]
        for p in mk.points:
            x,y = p.x, p.y
            th  = -math.degrees(math.atan2(y, x))
            r   = max(math.hypot(x,y), 0.3)
            half= 90.0 if self.base_r>=r else math.degrees(math.asin(self.base_r/r))
            inv = 1.0/r
            for i,(be,bh) in enumerate(zip(self.BIN_EDGES[:-1], self.BIN_EDGES[1:])):
                if bh >= th-half and be <= th+half:
                    h[i] = max(h[i], inv)
            obs.append((x,y,self.base_r))
        self.histogram = gaussian_filter1d(h, self.GAUSS_SIG)
        self.obstacles = obs
    # ╰────────────────────────────────────────────────────╯

    # ── 메인 루프 : 50 ms 주기 ──────────────────────────
    def main_loop(self):
        # 장애물 타임아웃
        if time.time() - self.last_marker_t > self.MARKER_TOUT:
            self.obstacles = []; self.histogram[:] = 0

        # MPC (장애물 있을 때만)
        if self.obstacles:
            _, path = self.run_mpc()
        else:
            path = self.gps_pts

        # Pure-Pursuit
        norm, la = self.pure_pursuit(path)
        self.steer_pub.publish(Float32(data=float(norm)))

        if self.viz:
            self.draw_cv(path, la)

    # ── MPC ----------------------------------------------------------------
    def run_mpc(self):
        best_deg, best_cost, best_path = 0.0, float('inf'), []
        for deg in np.arange(-self.max_deg, self.max_deg+1e-6, 2.0):
            path = self.sim_path(deg)
            cost = ( self.w_obs    * self.cost_obs(path) +
                     self.w_gps    * self.cost_gps(path) +
                     self.w_smooth * abs(deg - self.prev_opt_deg) )
            if cost < best_cost:
                best_deg, best_cost, best_path = deg, cost, path
        self.prev_opt_deg = best_deg
        return best_deg, best_path

    def sim_path(self, steer_deg):
        x=y=yaw=0.0; sr=math.radians(steer_deg); path=[]
        for _ in range(self.N_mpc):
            x   += self.speed_mpc*self.dt_mpc*math.cos(yaw)
            y   += self.speed_mpc*self.dt_mpc*math.sin(yaw)
            yaw += (self.speed_mpc/self.wheelbase)*math.tan(sr)*self.dt_mpc
            path.append((x,y))
        return path

    def cost_obs(self, path):
        h=self.histogram; c=0.0
        for px,py in path:
            th=-math.degrees(math.atan2(py,px))
            idx=np.clip(np.digitize(th,self.BIN_EDGES)-1,0,len(h)-1)
            c+=h[idx]
        return c

    def cost_gps(self, path):
        if not self.gps_pts: return 0.0
        arr=np.asarray(self.gps_pts)
        return sum(np.sqrt(np.min((arr[:,0]-px)**2+(arr[:,1]-py)**2))
                   for px,py in path)

    # ── Pure-Pursuit --------------------------------------------------------
    def pure_pursuit(self, path):
        if not path: return 0.0,None
        la=next((p for p in path if p[0]>=0 and math.hypot(*p)>=self.Ld_pp),
                path[-1])
        fwd,lat=la
        alpha = math.atan2(lat,fwd)
        delta = math.atan2(2*self.wheelbase*math.sin(alpha), self.Ld_pp)
        norm  = self.sign*float(np.clip(delta/math.radians(self.norm_deg),-1,1))
        return norm, la

    # ── 시각화 --------------------------------------------------------------
    def draw_cv(self, path, la):
        img=np.zeros((self.IMG_SZ,self.IMG_SZ,3),np.uint8)
        
        # GPS (파랑)
        if len(self.gps_pts)>1:
            cv2.polylines(img,[np.array([self.to_px(p) for p in self.gps_pts],np.int32)],
                          False,(255,0,0),1)
        # Obstacles (빨강)
        for x,y,r in self.obstacles:
            cv2.circle(img,self.to_px((x,y)),int(r*self.SCALE_PX),(0,0,255),2)
        # Path (노랑)
        if len(path)>1:
            cv2.polylines(img,[np.array([self.to_px(p) for p in path],np.int32)],
                          False,(0,255,255),2)
        # Look-ahead (초록)
        if la: cv2.circle(img,self.to_px(la),6,(0,255,0),-1)
        cv2.imshow("Predicted Path",img); cv2.waitKey(1)

    def to_px(self,xy):
        x,y=xy; b=(90+math.degrees(math.atan2(y,x)))%360
        d=math.hypot(x,y)*self.SCALE_PX; a=math.radians(b)
        return (int(self.CXY[0]+d*math.cos(a)),
                int(self.CXY[1]-d*math.sin(a)))

# ── main ───────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args); node=MPCPPNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node.viz: cv2.destroyAllWindows()
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
