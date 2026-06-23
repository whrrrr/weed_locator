#ifndef CONFIG_H_
#define CONFIG_H_

// 串口波特率
#define BAUD 115200

// 机器人臂长度
//#define SHANK_LENGTH 140.0
#define LOW_SHANK_LENGTH 140.0
#define HIGH_SHANK_LENGTH 160.0
#define END_EFFECTOR_OFFSET 50.0 // 从上臂轴承到末端执行器中点的长度（单位：毫米）

// 初始插值设置
// 初始 X、Y、Z 形成 90 度垂直下臂和水平上臂
#define INITIAL_X 0.0 // 笛卡尔坐标 X
#define INITIAL_Y 0.0 // 笛卡尔坐标 Y
#define INITIAL_Z -140.0 // 笛卡尔坐标 Z

#define INITIAL_E0 0.0 // 导轨步进电机的终止位置

// 校准归零步数，以达到所需的初始 X、Y、Z 位置
#define X_HOME_STEPS 0 //765 //860 // 从 X 终止开关到初始 X、Y、Z 位置的步数（上臂）
#define Y_HOME_STEPS 0 //1940 // 从 Y 终止开关到初始 X、Y、Z 位置的步数（下臂）
#define Z_HOME_STEPS 0 // 从 Z 终止开关到初始 X、Y、Z 位置的步数（旋转中心）
#define E0_HOME_STEPS 0 // 从 E0 终止开关到初始 E0 的步数

// 归零设置：
#define HOME_X_STEPPER true // 如果安装了终止开关，则为 "true"
#define HOME_Y_STEPPER true // 如果安装了终止开关，则为 "true"
#define HOME_Z_STEPPER true // 如果安装了终止开关，则为 "true"
#define HOME_E0_STEPPER false // 如果安装了终止开关，则为 "true"
#define HOME_ON_BOOT false // 如果开机时需要归零，则为 "true"
#define HOME_DWELL 1400 // 增加此值以减慢归零速度

// 步进电机设置：
#define MICROSTEPS 16 // RAMPS1.4 上的微步配置
#define STEPS_PER_REV 200 // NEMA17 每转的步数
#define INVERSE_X_STEPPER false // 如果步进电机反向运动，请修改此设置
#define INVERSE_Y_STEPPER false // 如果步进电机反向运动，请修改此设置
#define INVERSE_Z_STEPPER false // 如果步进电机反向运动，请修改此设置
#define INVERSE_E0_STEPPER false // 如果步进电机反向运动，请修改此设置

//

#define RAIL false // 如果 E0 电机作为底座导轨使用，请设置为 true；如果不使用，请设置为 false
#define STEPS_PER_MM_RAIL 80.0 // 导轨电机的每毫米步数
#define RAIL_LENGTH 200.0 // 导轨的最大长度（单位：毫米）

// 终止开关设置：
#define X_MIN_INPUT 1 // 开关激活时的输出值
#define Y_MIN_INPUT 1 // 开关激活时的输出值
#define Z_MIN_INPUT 1 // 开关激活时的输出值
#define E0_MIN_INPUT 0 // 开关激活时的输出值

// 传动比设置
#define MOTOR_GEAR_TEETH 1.0    // 20.0  齿轮
#define MAIN_GEAR_TEETH  4.5    // 90.0  齿轮

// 设备设置
#define LASER false // 12V 激光连接到 LASER_PIN
#define PUMP false // 12V 空气泵连接到 PUMP_PIN
#define FAN_DELAY 120 // 风扇延迟开启的时间（单位：秒）

// 夹爪设置
#define GRIPPER 1 // 使用的夹爪电机
        // 0: 28BYJ-48 微步电机
        // 1: 9G 伺服电机或 MG996 伺服电机等效
// 28BYJ 夹爪设置
#define BYJ_GRIP_STEPS 1200 // FTOBLER: 1200
// 伺服夹爪设置
#define SERVO_GRIP_DEGREE 90.0
#define SERVO_UNGRIP_DEGREE 0.0

// 命令队列设置
#define QUEUE_SIZE 10               //15

// 打印回复设置
#define PRINT_REPLY true // "true" 在处理完命令后打印消息
#define PRINT_REPLY_MSG "ok" // 发送的消息，供用户与其他软件进行后处理

// 速度曲线设置
#define SPEED_PROFILE 2 // 以下选项
//0: 平滑速度曲线（每次运动的恒定速度，适合实时控制软件）
//1: 弧度近似（轻微的钟形加速和减速）
//2: 余弦近似（从 0 开始加速，减速到 0，适合预设命令的运动）

// 日志设置
#define LOG_LEVEL 2
//0: 错误
//1: 信息
//2: 调试

// 移动限制参数
#define Z_MIN -320.0
#define Z_MAX -140.0
#define SHANKS_MIN_ANGLE_COS 0.791436948
#define SHANKS_MAX_ANGLE_COS -0.774944489
#define R_MIN (sqrt((sq(LOW_SHANK_LENGTH) + sq(HIGH_SHANK_LENGTH)) - (2*LOW_SHANK_LENGTH*HIGH_SHANK_LENGTH*SHANKS_MIN_ANGLE_COS) ))
#define R_MAX (sqrt((sq(LOW_SHANK_LENGTH) + sq(HIGH_SHANK_LENGTH)) - (2*LOW_SHANK_LENGTH*HIGH_SHANK_LENGTH*SHANKS_MAX_ANGLE_COS) ))

#endif
