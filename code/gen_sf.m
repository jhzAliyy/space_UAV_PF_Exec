% gen_sf.m — QuaDRiGa qd_sos 空间相关阴影场生成（引擎原生路径）
% 用法（Octave）：
%   gen_sf <quadriga_src路径> <cells.csv> <m_lo> <m_hi> <dcorr_m> <sigma_dB> <seed> <out.csv>
% cells.csv: C×3 (x,y,z) 网格坐标；用户 m∈[m_lo, m_hi) 各一个独立 SOS 实现
% 输出 out.csv: (m_hi−m_lo)×C 的 SF [dB] = sigma × N(0,1) 空间相关场
% 参数权威源：QuaDRiGa 官方 conf（3GPP 38.901 UMa/RMa SF_sigma/SF_lambda）

function gen_sf(qd_path, cells_csv, m_lo, m_hi, dcorr, sigma, seed, out_csv)

addpath(qd_path);

C = csvread(cells_csv);              % C×3
n_c = size(C, 1);
n_m = m_hi - m_lo;

rand('state', mod(seed + m_lo * 7919, 2^31 - 1));   % 分块确定性种子

s = qd_sos.randn(dcorr * ones(1, n_m), C', []);     % [n_m × n_c]

csvwrite(out_csv, sigma * s);

end
