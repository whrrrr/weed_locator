# SO101 平滑运动调试交接文档

日期：2026-07-08

这份文档只整理 SO101 / LeRobot 机械臂这条线，和“语音手套”项目分开。语音手套仍然保留在原来的对话/上下文里继续推进；本文件用于把 2026-07-07 这一天临时插进来的 SO101 舵机平滑调试完整收口。

## 背景

昨天的问题是：SO101 总线舵机运动时看起来“一步一步走”、不够丝滑。我们想确认：

```text
1. 是不是命令点太稀疏？
2. 是不是刷新率不够？
3. 是不是舵机/机械本身有负载、齿隙、死区、控制环抖动？
4. 如果要尽量丝滑，这套硬件大概要慢到什么程度？
```

最开始的判断是：总线舵机可以通过“轨迹插值 + S 曲线加减速”改善，但硬件本身有极限，尤其是带负载关节。

## 关键结论

整体结论：

```text
软件插值能让运动更连续，但这套硬件在负载关节上要想完全不抖，需要非常慢。
第二、第三关节比较稳定的速度量级约为 0.5 deg/s。
这个速度很丝滑，但肉眼看起来会很慢，实际使用需要在“丝滑”和“可用速度”之间折中。
```

更具体：

```text
shoulder_pan：
  第一个左右转向舵机负载小，问题不明显。

shoulder_lift：
  主要受负载/动态响应限制。速度上来会抖。
  2 deg / 4 s 比较顺，约 0.5 deg/s。

elbow_flex：
  小角度有明显死区/齿隙，1 deg 动作不明显。
  2 deg 才比较可见。
  提高刷新率到 20 Hz 并不会更顺，反而更容易看出一格一格。
```

所以后续运动学节点不是单纯提高 `rate_hz`，而是加入速度限制和 S 曲线。

## 测试脚本

新增脚本：

```text
src/weed_locator/scripts/lerobot_servo_smoothness_probe.py
```

用途：

```text
以当前关节角度为 center，只测试一个关节的小幅往返运动：
center -> center + amplitude -> center

通过改变：
  amplitude-deg
  duration
  rate-hz
  profile

观察舵机是否：
  一格一格
  抖
  滞后
  过冲
  几乎不动
```

它默认是手动确认模式：每段动作前会停住，让操作者确认无干涉后按 Enter，避免机械臂自动扫到危险位置。

常用命令：

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

./src/weed_locator/scripts/lerobot_servo_smoothness_probe.py \
  --joint shoulder_lift \
  --amplitude-deg 2 \
  --duration 4.0 \
  --rate-hz 10 \
  --profile smootherstep
```

可选关节：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
```

参数含义：

```text
--amplitude-deg 2
  当前角度向一个方向偏移 2 度，然后回到当前角度。

--duration 4.0
  center -> edge 用 4 秒，edge -> center 也用 4 秒。

--rate-hz 10
  每秒发 10 次小目标。

--profile smootherstep
  使用更柔和的 S 曲线，起步和停止都慢。
```

注意：

```text
不要一开始使用 --auto。
不要一开始使用 --sweep。
先小幅度、手动确认。
```

## 实测记录

原始简表也保存在：

```text
/home/whr/cc_ws/tros_ws/calibration_targets/so101_servo_smoothness_notes.md
```

### 关节 2：shoulder_lift

测试姿态附近：

```text
center ~= 60.3 deg
```

结果：

```text
amplitude=1 deg, duration=2.0 s, rate=10 Hz, profile=smootherstep
  -> smooth

amplitude=2 deg, duration=2.0 s, rate=10 Hz, profile=smootherstep
  -> starts to feel stuck/steppy

amplitude=2 deg, duration=4.0 s, rate=10 Hz, profile=smootherstep
  -> smooth

amplitude=2 deg, duration=2.0 s, rate=20 Hz, profile=smootherstep
  -> shakes
```

解释：

```text
A：2 deg / 4 s / 10 Hz
  平均速度 0.5 deg/s，约 40 个目标点，每步平均 0.05 deg。
  结果挺顺。

B：2 deg / 2 s / 20 Hz
  平均速度 1.0 deg/s，也是约 40 个目标点，每步平均 0.05 deg。
  结果开始抖。
```

因此问题不是点数不够，而是负载下的速度/动态响应极限。

推荐保守值：

```text
shoulder_lift max speed ~= 0.5 deg/s
profile = smootherstep
rate = 10 Hz
```

### 关节 3：elbow_flex

测试姿态附近：

```text
center ~= 42 deg
```

结果：

```text
amplitude=1 deg, duration=2.0 s, rate=10 Hz, profile=smootherstep
  -> barely visible / feels like it does not move

amplitude=2 deg, duration=2.0 s, rate=10 Hz, profile=smootherstep
  -> visible movement

amplitude=2 deg, duration=4.0 s, rate=10 Hz, profile=smootherstep
  -> no obvious shake; slower and acceptable

amplitude=2 deg, duration=2.0 s, rate=20 Hz, profile=smootherstep
  -> no obvious shake, but visibly faster and more step-by-step
```

解释：

```text
这个关节在当前姿态附近很可能有 1 deg 量级的死区/齿隙/静摩擦。
1 deg 太小，动作不明显。
2 deg 才比较可见。
提高到 20 Hz 不会自动变顺，反而因为同样 2 deg 被压缩到 2 秒完成，格子感更明显。
```

推荐保守值：

```text
elbow_flex minimum useful move ~= 2 deg
elbow_flex comfortable speed ~= 0.5 deg/s
profile = smootherstep
rate = 10 Hz
```

## 运动学键盘节点改动

改动文件：

```text
src/weed_locator/scripts/lerobot_no_elbow_ik_keyboard_jog.py
```

原逻辑：

```text
按一下键
IK 算出最终角度
直接把最终角度一次性发给舵机
```

新逻辑：

```text
按一下键
IK 算出最终角度
根据每个关节速度限制，自动计算运动时间
用 smootherstep S 曲线分成中间目标
按 10 Hz 连续发送给舵机
```

新增默认参数：

```text
smooth_execute = true
smooth_rate_hz = 10.0
smooth_profile = smootherstep
min_smooth_duration_sec = 0.25
joint_speed_limits_deg_s = shoulder_pan:2.0,shoulder_lift:0.5,wrist_flex:0.5
```

注意 no-elbow 结构：

```text
显示/URDF joint: wrist_flex
实际硬件第三舵机: elbow_flex
```

所以这里 `wrist_flex:0.5` 实际也是在限制第三个物理舵机。

启动命令：

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

./src/weed_locator/scripts/lerobot_no_elbow_ik_keyboard_jog.py
```

显式指定平滑参数：

```bash
./src/weed_locator/scripts/lerobot_no_elbow_ik_keyboard_jog.py \
  --ros-args \
  -p smooth_execute:=true \
  -p smooth_rate_hz:=10.0 \
  -p smooth_profile:=smootherstep \
  -p joint_speed_limits_deg_s:=shoulder_pan:2.0,shoulder_lift:0.5,wrist_flex:0.5
```

如果觉得太慢，可以试用折中参数：

```bash
./src/weed_locator/scripts/lerobot_no_elbow_ik_keyboard_jog.py \
  --ros-args \
  -p smooth_execute:=true \
  -p smooth_rate_hz:=10.0 \
  -p smooth_profile:=smootherstep \
  -p joint_speed_limits_deg_s:=shoulder_pan:3.0,shoulder_lift:0.8,wrist_flex:0.8
```

键盘控制：

```text
w/s: +x/-x
a/d: +y/-y
r/f: +z/-z
o/p: 第四辅助关节 +/-
c: 同步 command cache 到当前真实关节
+/-: 增大/减小步长
q: 退出
```

运行时如果平滑轨迹生效，会看到日志类似：

```text
smooth trajectory: duration=..., rate=10.0Hz, steps=..., delta_deg=[...]
```

## SO101 bridge 改动

改动文件：

```text
src/weed_locator/scripts/lerobot_so101_bridge.py
```

新增/整理点：

```text
1. 支持 hardware_joint_map
   用于逻辑关节和真实硬件舵机名之间临时映射。

2. WriteJoints 支持 NaN 跳过
   这样可以只写某一个关节，其它关节保持不动。

3. P_Coefficient 设置支持逻辑关节 -> 硬件关节映射
   方便以后调舵机 P 增益。
```

这些改动是平滑测试脚本和 no-elbow 节点能稳定工作的重要基础。

## CMake 安装项

改动文件：

```text
src/weed_locator/CMakeLists.txt
```

新增安装：

```text
scripts/lerobot_no_elbow_ik_keyboard_jog.py
```

这样以后 colcon 安装后也可以通过 ROS 包安装路径找到它。

## Git 状态

SO101 相关改动已提交到 `src/weed_locator` 子仓库并 push。

提交：

```text
014d048 Add smooth SO101 jog probing tools
```

包含文件：

```text
CMakeLists.txt
scripts/lerobot_so101_bridge.py
scripts/lerobot_no_elbow_ik_keyboard_jog.py
scripts/lerobot_servo_smoothness_probe.py
```

push 目标：

```text
origin master
ssh://git@ssh.github.com:443/whrrrr/weed_locator.git
```

当时仓库里还有两个未提交改动，没有混入这次 SO101 平滑提交：

```text
config/SO101/so101_joint_display_offsets.yaml
scripts/delta_visual_pick_demo.py
```

## 当前问题与后续建议

### 1. 速度太慢

0.5 deg/s 很稳，但实际操作很慢。后面可以尝试折中：

```text
shoulder_lift: 0.8 -> 1.0 deg/s
wrist_flex/第三物理舵机: 0.8 -> 1.0 deg/s
shoulder_pan: 2.0 -> 3.0 deg/s
```

判断标准：

```text
如果开始明显抖/卡，就降回上一档。
如果只是轻微格子感但可接受，可以作为实际操作参数。
```

### 2. 第三关节小动作不明显

第三关节 1 deg 动作不明显，可能被齿隙/死区吃掉。实际控制时不要期望每次 0.1 deg 的微小动作都能在末端表现出来。

### 3. 提高刷新率不一定有用

20 Hz 不必然更顺。对这些舵机来说，轨迹速度和机械响应更关键。盲目提高 `smooth_rate_hz` 可能只会让目标更密，但舵机跟踪更紧，抖动更明显。

### 4. 以后可以继续调 P 增益

如果要进一步提升速度又不抖，可以尝试：

```text
降低 P_Coefficient
做重力补偿
机械消隙
减小负载力臂
更强舵机
```

现在已经在 bridge 里保留了 `motor_p_coefficients` 和映射支持，可以后续继续实验。

## 和语音手套项目的边界

这份文档属于 SO101 机械臂调试，不属于语音手套。

语音手套继续沿原来的主线：

```text
语音学/音系学依据
振动触觉 vs 电触觉
TENS/EMS 电刺激编码
银纤维布电极手套制作
左右手声母/韵母编码
```

后续如果继续本对话，默认回到语音手套，不再把 SO101 的运动学细节混进来。
