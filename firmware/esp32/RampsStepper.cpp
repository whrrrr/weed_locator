#include "RampsStepper.h"
#include "config.h"

#include <Arduino.h>


// 构造函数
RampsStepper::RampsStepper(int aStepPin, int aDirPin, int aEnablePin, bool aInverse) {
  // 1. 设定减速比，使用齿轮比和每转步数计算步进电机的转角步数
  setReductionRatio(MAIN_GEAR_TEETH / MOTOR_GEAR_TEETH, MICROSTEPS * STEPS_PER_REV);

  stepPin = aStepPin;  // 步进引脚
  dirPin = aDirPin;    // 方向引脚
  enablePin = aEnablePin;  // 使能引脚
  inverse = aInverse;  // 是否反向控制电机（true 为反向，false 为正向）

  stepperStepPosition = 0;  // 当前步进电机的步进位置
  stepperStepTargetPosition = 0;  // 目标步进位置

  // 设置引脚模式为输出
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enablePin, OUTPUT);

  enable(false);  // 初始时禁用电机
}

// 启用或禁用步进电机
void RampsStepper::enable(bool value) {
  digitalWrite(enablePin, value);  // 根据 value 启用或禁用电机
}

// 判断电机是否已到达目标位置
bool RampsStepper::isOnPosition() const {
  return stepperStepPosition == stepperStepTargetPosition;
}

// 获取当前步进位置
int RampsStepper::getPosition() const {
  return stepperStepPosition;
}

// 设置当前步进电机的位置
void RampsStepper::setPosition(int value) {
  stepperStepPosition = value;
  stepperStepTargetPosition = value;  // 更新目标位置
}

// 设置目标步进位置
void RampsStepper::stepToPosition(int value) {
  stepperStepTargetPosition = value;
}

// 设置目标位置（单位：毫米）
void RampsStepper::stepToPositionMM(float mm, float steps_per_mm) {
  stepperStepTargetPosition = mm * steps_per_mm;  // 将毫米转化为步进位置
}

// 设置相对目标位置
void RampsStepper::stepRelative(int value) {
  value += stepperStepPosition;  // 在当前步进位置基础上增加相对位置
  stepToPosition(value);  // 设置新的目标位置
}

// 获取当前步进位置的角度（单位：弧度）
float RampsStepper::getPositionRad() const {
  return stepperStepPosition / radToStepFactor;  // 将步进位置转换为弧度
}

// 设置步进电机的位置（单位：弧度）
void RampsStepper::setPositionRad(float rad) {
  setPosition(rad * radToStepFactor);  // 将弧度转化为步进位置并设置
}

// 设置目标位置（单位：弧度）
void RampsStepper::stepToPositionRad(float rad) {
  stepperStepTargetPosition = rad * radToStepFactor;  // 将弧度转化为步进位置
}

// 设置相对目标位置（单位：弧度）
void RampsStepper::stepRelativeRad(float rad) {
  stepRelative(rad * radToStepFactor);  // 将弧度转化为步进位置并设置相对目标
}

// 更新步进电机状态，控制电机按步进位置前进或后退
void RampsStepper::update() {
  // 如果当前步进位置大于目标位置，电机需要后退

  while (stepperStepTargetPosition < stepperStepPosition) {
    digitalWrite(dirPin, !inverse);  // 设置反向（根据 inverse 设定）
    digitalWrite(stepPin, HIGH);  // 发送一步信号
    digitalWrite(stepPin, LOW);  // 清除一步信号
    stepperStepPosition--;  // 更新当前位置
  }

  // 如果当前步进位置小于目标位置，电机需要前进
  while (stepperStepTargetPosition > stepperStepPosition) {
    digitalWrite(dirPin, inverse);  // 设置正向（根据 inverse 设定）
    digitalWrite(stepPin, HIGH);  // 发送一步信号
    digitalWrite(stepPin, LOW);  // 清除一步信号
    stepperStepPosition++;  // 更新当前位置
  }
}

// 设定减速比，计算每个步进的弧度数
void RampsStepper::setReductionRatio(float gearRatio, int stepsPerRev) {
  radToStepFactor = gearRatio * stepsPerRev / 2 / PI;  // 根据齿轮比和每转步数计算弧度到步进数的转换因子
};
