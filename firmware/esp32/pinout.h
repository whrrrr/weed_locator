#ifndef PINOUT_H_
#define PINOUT_H_


// --------- 3个步进点击对应的3轴电路板上的引脚 ---------
#define X_STEP_PIN         14
#define X_DIR_PIN          27
#define X_ENABLE_PIN       12
#define X_MIN_PIN          36  // 最小位置的限位器引脚
#define X_MAX_PIN          4

#define Y_STEP_PIN         26
#define Y_DIR_PIN          25
#define Y_ENABLE_PIN       12
#define Y_MIN_PIN          39 // 最小位置的限位器引脚
#define Y_MAX_PIN          4

#define Z_STEP_PIN         33
#define Z_DIR_PIN          32
#define Z_ENABLE_PIN       12
#define Z_MIN_PIN          34 // 最小位置的限位器引脚
#define Z_MAX_PIN          4

#define E0_STEP_PIN        19
#define E0_DIR_PIN         18
#define E0_ENABLE_PIN      12
#define E0_MIN_PIN         4

#define E1_STEP_PIN        -1
#define E1_DIR_PIN         -1
#define E1_ENABLE_PIN      -1

#define BYJ_PIN_0          -1
#define BYJ_PIN_1          -1
#define BYJ_PIN_2          -1
#define BYJ_PIN_3          -1

// #define SERVO_PIN          A4

#define PUMP_PIN            21  // 气泵
#define SOLENOID_VALUE_PIN  16  //  电磁阀
#define LASER_PIN          -1
#define LED_PIN            -1

// #define PUMP1_PIN           A0
// #define PUMP2_PIN           A1
// #define PUMP3_PIN           A2
// #define PUMP4_PIN           A3

#define SDPOWER            -1
#define SDSS               -1

#define FAN_PIN            -1

#define PS_ON_PIN          -1
#define KILL_PIN           -1

//#define HEATER_0_PIN       10
//#define HEATER_1_PIN        8
#define TEMP_0_PIN         -1   // ANALOG NUMBERING
#define TEMP_1_PIN         -1   // ANALOG NUMBERING

#endif
