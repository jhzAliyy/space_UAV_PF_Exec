// crowd_sfm.cpp —— 人群社会力引擎封装（2026-10-27，句柄版）
// 本体：deps/lightsfm（BSD-3，Helbing 社会力，公式级原样调用）；
// 加速：均匀空间网格近邻枚举（力公式一行不动，只换邻居来源）；
// 多实例：init 返回句柄，step/gather 带句柄——环境实例互不串台；
// 时间轴：dt 由调用方传入（人群步长独立于无人机物理步长，分钟级重构口径）。
#include "sfm.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

// 人群步进线程数（OpenMP）。默认 1 = 历史口径 bit 级不变；
// PF_CROWD_THREADS>1 启用多核：每 agent 独立计算、邻居集为快照拷贝、
// SocialForceModel 全方法 const 无内部可变状态——多线程结果与单线程
// 逐位一致（同 seed 同轨迹）。
static inline int pf_crowd_nt(void) {
  const char* e = getenv("PF_CROWD_THREADS");
  int nt = e ? atoi(e) : 1;
  return nt > 0 ? nt : 1;
}

struct CrowdState {
  std::vector<sfm::Agent> agents;
  double fw;
  double fh;
  double cell;
};

extern "C" {

// 初始化：返回句柄（非空）；失败返回 NULL。
// 基线流动=每人绕自身初始位置半径 roam_r 的圆走 4 个途经点（cyclic）。
void* crowd_init(int n, const double* pos, const double* v_des,
                 double field_w, double field_h,
                 double roam_fast, double roam_slow,
                 double goal_r_fast, double goal_r_slow, double cell) {
  CrowdState* s = new CrowdState();
  s->agents.resize(n);
  s->fw = field_w;
  s->fh = field_h;
  s->cell = cell > 0.0 ? cell : 3.0;
  for (int i = 0; i < n; i++) {
    sfm::Agent& a = s->agents[i];
    a.position = utils::Vector2d(pos[2 * i], pos[2 * i + 1]);
    a.velocity = utils::Vector2d(0.0, 0.0);        // 静止起步
    a.yaw = utils::Angle::fromRadian(0.0);
    a.desiredVelocity = v_des[i];
    a.radius = 0.35;                                // lightsfm 人体默认
    a.params = sfm::Parameters();                   // 文献默认力学参数原样
    a.groupId = -1;
    a.id = i;
    a.cyclicGoals = true;
    a.teleoperated = false;
    double roam = a.desiredVelocity > 0.9 ? roam_fast : roam_slow;
    double goal_r = a.desiredVelocity > 0.9 ? goal_r_fast : goal_r_slow;
    double x0 = pos[2 * i];
    double y0 = pos[2 * i + 1];
    for (int k = 0; k < 4; k++) {                   // 圆上四途经点循环
      double th = 2.0 * M_PI * k / 4.0;
      double gx = x0 + roam * std::cos(th);
      double gy = y0 + roam * std::sin(th);
      gx = gx < 0.0 ? 0.0 : (gx > s->fw ? s->fw : gx);
      gy = gy < 0.0 ? 0.0 : (gy > s->fh ? s->fh : gy);
      sfm::Goal g;
      g.center = utils::Vector2d(gx, gy);
      g.radius = goal_r;
      a.goals.push_back(g);
    }
  }
  return s;
}

// 一步推进：网格分桶 -> 逐人力计算+积分（lightsfm 原码）-> 场地钳制 ->
// 写 pos_out[2n]。
int crowd_step(void* handle, double dt, double* pos_out) {
  CrowdState* s = (CrowdState*)handle;
  int n = (int)s->agents.size();
  if (n == 0 || dt <= 0.0) {
    return -1;
  }
  int ncx = (int)std::floor(s->fw / s->cell) + 1;
  int ncy = (int)std::floor(s->fh / s->cell) + 1;
  std::vector<std::vector<int>> cell_of((size_t)ncx * (size_t)ncy);
  for (int i = 0; i < n; i++) {
    int cx = (int)std::floor(s->agents[i].position.getX() / s->cell);
    int cy = (int)std::floor(s->agents[i].position.getY() / s->cell);
    cx = cx < 0 ? 0 : (cx >= ncx ? ncx - 1 : cx);
    cy = cy < 0 ? 0 : (cy >= ncy ? ncy - 1 : cy);
    cell_of[(size_t)cy * (size_t)ncx + (size_t)cx].push_back(i);
  }
  sfm::SocialForceModel& model = sfm::SocialForceModel::getInstance();
#pragma omp parallel num_threads(pf_crowd_nt())
  {
    std::vector<sfm::Agent> near_agents;            // 线程私有邻居集
#pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
      int cx = (int)std::floor(s->agents[i].position.getX() / s->cell);
      int cy = (int)std::floor(s->agents[i].position.getY() / s->cell);
      near_agents.clear();
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          int nx = cx + dx;
          int ny = cy + dy;
          if (nx < 0 || nx >= ncx || ny < 0 || ny >= ncy) {
            continue;
          }
          const std::vector<int>& bucket =
              cell_of[(size_t)ny * (size_t)ncx + (size_t)nx];
          for (size_t b = 0; b < bucket.size(); b++) {
            near_agents.push_back(s->agents[bucket[b]]);
          }
        }
      }
      model.computeForces(s->agents[i], near_agents, NULL);
      model.updatePosition(s->agents[i], dt);
      double px = s->agents[i].position.getX();
      double py = s->agents[i].position.getY();
      px = px < 0.0 ? 0.0 : (px > s->fw ? s->fw : px);
      py = py < 0.0 ? 0.0 : (py > s->fh ? s->fh : py);
      s->agents[i].position = utils::Vector2d(px, py);
    }
  }
  for (int i = 0; i < n; i++) {
    pos_out[2 * i] = s->agents[i].position.getX();
    pos_out[2 * i + 1] = s->agents[i].position.getY();
  }
  return 0;
}

// 突发聚集：k 个聚集中心，每用户就近选择，目标重设（到达驻留）；
// 涌向速度 v_rush 统一设定（"突然集中"=目标切换+提速赶场）。
int crowd_gather(void* handle, const double* centers, int k,
                 double gather_r, double v_rush) {
  CrowdState* s = (CrowdState*)handle;
  int n = (int)s->agents.size();
  if (n == 0 || k <= 0) {
    return -1;
  }
  for (int i = 0; i < n; i++) {
    sfm::Agent& a = s->agents[i];
    double best_d2 = 1e30;
    int best = 0;
    for (int c = 0; c < k; c++) {
      double dx = a.position.getX() - centers[2 * c];
      double dy = a.position.getY() - centers[2 * c + 1];
      double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = c;
      }
    }
    a.goals.clear();
    a.cyclicGoals = false;
    a.desiredVelocity = v_rush;
    sfm::Goal g;
    g.center = utils::Vector2d(centers[2 * best], centers[2 * best + 1]);
    g.radius = gather_r;
    a.goals.push_back(g);
  }
  return 0;
}

int crowd_free(void* handle) {
  CrowdState* s = (CrowdState*)handle;
  if (s != NULL) {
    delete s;
  }
  return 0;
}

}  // extern "C"
